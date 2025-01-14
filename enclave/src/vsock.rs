use anyhow::{Context, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
use tracing::{error, info, instrument};

use nsm_io::{Request as NsmRequest, Response as NsmResponse};
use serde_bytes::ByteBuf;
use sha2::{Digest, Sha384};

use llama_runner::{LlamaRunner, LlamaConfig};

/// The size of each chunk to read/write (10 MB).
pub const BUFFER_SIZE: usize = 10 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub enum Operation {
    SendFile,
    EofFile,
    Prompt,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub op: Operation,
    pub data: Vec<u8>,
}
fn generate_attestation(
    input_prompt: &str,
    output_prompt: &str,
    model_id: &str,
) -> Result<Vec<u8>> {
    // Calculate hashes
    let input_hash = Sha384::digest(input_prompt.as_bytes());
    let output_hash = Sha384::digest(output_prompt.as_bytes());
    let model_hash = Sha384::digest(model_id.as_bytes());

    // Concatenate hashes into user_data
    let mut user_data = Vec::new();
    user_data.extend_from_slice(&input_hash);
    user_data.extend_from_slice(&output_hash);
    user_data.extend_from_slice(&model_hash);

    // Initialize NSM
    let nsm_fd = nsm_driver::nsm_init();
    if nsm_fd < 0 {
        return Err(anyhow::anyhow!("Failed to initialize NSM"));
    }

    // Create attestation request
    let request = NsmRequest::Attestation {
        public_key: None,
        user_data: Some(ByteBuf::from(user_data)),
        nonce: None,
    };

    // Process the attestation request
    let response = nsm_driver::nsm_process_request(nsm_fd, request);

    // Match against the correct variants
    let result = match response {
        NsmResponse::Attestation { document, .. } => Ok(document),
        NsmResponse::Error(error_code) => Err(anyhow::anyhow!("NSM Error: {:?}", error_code)),
        _ => Err(anyhow::anyhow!("Unexpected NSM response")),
    };

    // Clean up
    nsm_driver::nsm_exit(nsm_fd);

    result
}

// ----------------------------------
// Shared logic for reading/writing
// ----------------------------------

/// Read a `Message` from the stream using a 4-byte length prefix before the bincode payload.
#[instrument(skip(stream))]
pub async fn read_message(stream: &mut VsockStream) -> Result<Message> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let msg_len = u32::from_be_bytes(len_buf) as usize;

    let mut buf = vec![0u8; msg_len];
    stream.read_exact(&mut buf).await?;

    let msg: Message = bincode::deserialize(&buf)?;
    Ok(msg)
}

/// Write a `Message` to the stream using a 4-byte length prefix before the bincode payload.
#[instrument(skip(stream, msg))]
pub async fn write_message(stream: &mut VsockStream, msg: &Message) -> Result<()> {
    let encoded = bincode::serialize(msg)?;
    let len_bytes = (encoded.len() as u32).to_be_bytes();

    stream.write_all(&len_bytes).await?;
    stream.write_all(&encoded).await?;

    Ok(())
}

// ----------------------------------
// Server-side logic
// ----------------------------------

/// Listens on the specified port for incoming `VsockStream` connections.
/// Spawns a new task to handle each connection.
#[instrument]
pub async fn run_server(port: u32) -> Result<()> {
    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr).context("Unable to bind Virtio listener")?;
    info!("Listening for connections on port: {}", port);

    let mut incoming = listener.incoming();
    while let Some(stream_result) = incoming.next().await {
        match stream_result {
            Ok(mut stream) => {
                info!("Got connection from client");
                tokio::spawn(async move {
                    if let Err(e) = handle_incoming_messages(&mut stream).await {
                        error!("Error handling client: {:?}", e);
                    }
                });
            }
            Err(e) => {
                error!("Error accepting connection: {:?}", e);
            }
        }
    }

    Ok(())
}

/// Receives `Message` structs from a client.
/// - If `op == SendFile`, writes the data to `model.gguf`.
/// - If `op == EofFile`, stops receiving.
#[instrument(skip(stream))]
async fn handle_incoming_messages(stream: &mut VsockStream) -> Result<()> {
    let mut file = None;
    let mut pb = None;
    let mut total_received = 0;

    loop {
        let msg = match read_message(stream).await {
            Ok(m) => m,
            Err(e) => {
                error!("Error reading bincode message: {:?}", e);
                break;
            }
        };

        match msg.op {
            Operation::SendFile => {
                if file.is_none() {
                    file = Some(File::create("model.gguf")
                        .await
                        .context("Failed to create file 'model.gguf'")?);
                    
                    pb = Some(ProgressBar::new_spinner());
                    pb.as_ref().unwrap().set_style(
                        ProgressStyle::default_spinner()
                            .template("{spinner} [{elapsed_precise}] {msg}")
                            .unwrap(),
                    );
                    pb.as_ref().unwrap().set_message("Receiving data...");
                    pb.as_ref().unwrap().enable_steady_tick(Duration::from_millis(100));
                }

                if let Some(ref mut f) = file {
                    f.write_all(&msg.data).await?;
                    total_received += msg.data.len() as u64;
                    if let Some(ref pb) = pb {
                        pb.set_message(format!("Wrote {} bytes to 'model.gguf'", total_received));
                    }
                }
            }
            Operation::EofFile => {
                if let Some(ref pb) = pb {
                    pb.finish_with_message("File transfer complete. Received EOF marker.");
                }
                info!("Received end-of-file marker. Stopping file reception.");
                info!("File transfer complete.");

                break;
            }
            Operation::Prompt => {     
                let config = LlamaConfig::new("model.gguf");

                let mut runner = LlamaRunner::new(config);

                runner.load_model()?;
                info!("Loaded model.");

                println!("---- Streaming tokens for prompt");
                let prompt = match String::from_utf8(msg.data.clone()) {
                    Ok(p) => p,
                    Err(_) => "Default prompt.".to_string(),
                };
                
                let final_text_1 = runner.generate_stream(&prompt, |_token| {
                    // Here we just print again, but you could do something else,
                    // like sending tokens over a channel or a websocket.
                    // For example: ws.send(token).await? if you had an async context
                })?;
                println!("Final text #1:\n{final_text_1}");
                    
                // Perform attestation, TODO CS: make async
                match tokio::task::block_in_place(|| {
                    // TODO CS: these are just example parameterrs, think about the sensible choices
                    generate_attestation(&prompt, &final_text_1, "model-123")
                }) {
                    Ok(attestation_response) => {
                        info!("Attestation Response: {:?}", attestation_response);
                    }
                    Err(e) => {
                        error!("Failed to generate attestation: {:?}", e);
                    }
                }

                break;
            }
        }
    }

    Ok(())
}

// ----------------------------------
// Client-side logic
// ----------------------------------

/// Connects to a server at the provided CID/port and sends the specified file in chunks.
/// After fully sending the file, an `EofFile` message is transmitted.
#[instrument]
pub async fn run_client(port: u32, cid: u32, file_path: Option<&str>, prompt: Option<&str>) -> Result<()> {
    let addr = VsockAddr::new(cid, port);
    let mut stream = VsockStream::connect(addr)
        .await
        .context("Failed to connect to server")?;

    info!(
        "Connected to server at CID {}:PORT {}",
        tokio_vsock::VMADDR_CID_LOCAL,
        port
    );

    if file_path.is_some() {
        let file_path = file_path.unwrap();
        let mut file = File::open(file_path)
            .await
            .context("Failed to open file for reading")?;
    
        let file_metadata = tokio::fs::metadata(file_path).await?;
        let total_size = file_metadata.len();
        let mut total_sent = 0;
    
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                )
                .expect("Failed to set template for progress bar")
                .progress_chars("#>-"),
        );
        pb.set_message(format!("Transferring '{}'...", file_path));   

        let mut buf = vec![0u8; BUFFER_SIZE];

        loop {
            let len = file.read(&mut buf).await?;
            if len == 0 {
                // EOF => send EofFile
                let msg = Message {
                    op: Operation::EofFile,
                    data: vec![],
                };
                write_message(&mut stream, &msg).await?;
                pb.finish_with_message("File transfer complete. Sent EOF marker.");
                break;
            }

            let msg = Message {
                op: Operation::SendFile,
                data: buf[..len].to_vec(),
            };
            write_message(&mut stream, &msg).await?;

            total_sent += len as u64;
            pb.set_position(total_sent);
        }
    }

    if let Some(prompt) = prompt {
        let msg = Message {
            op: Operation::Prompt,
            data: prompt.as_bytes().to_vec(),
        };
        write_message(&mut stream, &msg).await?;
    }
    

    Ok(())
}