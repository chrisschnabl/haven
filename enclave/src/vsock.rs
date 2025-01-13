use anyhow::{Context, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
use tracing::{error, info, instrument};

/// The size of each chunk to read/write (10 MB).
pub const BUFFER_SIZE: usize = 10 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub enum Operation {
    SendFile,
    EofFile,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub op: Operation,
    pub data: Vec<u8>,
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
    let mut file = File::create("model.gguf")
        .await
        .context("Failed to create file 'model.gguf'")?;

    let mut total_received = 0;
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} [{elapsed_precise}] {msg}")
            .unwrap(),
    );
    pb.set_message("Receiving data...");
    pb.enable_steady_tick(Duration::from_millis(100));

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
                file.write_all(&msg.data).await?;
                total_received += msg.data.len() as u64;
                pb.set_message(format!("Wrote {} bytes to 'model.gguf'", total_received));
            }
            Operation::EofFile => {
                pb.finish_with_message("File transfer complete. Received EOF marker.");
                info!("Received end-of-file marker. Stopping file reception.");
                info!("File transfer complete.");
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
pub async fn run_client(port: u32, cid: u32, file_path: &str) -> Result<()> {
    let addr = VsockAddr::new(cid, port);
    let mut stream = VsockStream::connect(addr)
        .await
        .context("Failed to connect to server")?;

    info!(
        "Connected to server at CID {}:PORT {}",
        tokio_vsock::VMADDR_CID_LOCAL,
        port
    );

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

    Ok(())
}