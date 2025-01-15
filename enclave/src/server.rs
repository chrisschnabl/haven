use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockListener};
use tracing::{error, info, instrument};
use crate::vsock::{read_message, Operation};
use tokio::fs::File;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use tokio::io::{AsyncWriteExt};
use futures::StreamExt;
use tokio_vsock::VsockStream;
use llama_runner::LlamaConfig;
use llama_runner::LlamaRunner;
use crate::attestation::generate_attestation;

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
                
                let text = runner.generate_stream(&prompt, |_token| {
                    // TODO CS: stream token to client
                })?;
                println!("Output :\n{text}");
                    
                // Perform attestation, TODO CS: make async
                match tokio::task::block_in_place(|| {
                    // TODO CS: these are just example parameterrs, think about the sensible choices
                    generate_attestation(&prompt, &text, "model-123")
                }) {
                    Ok(attestation_response) => {
                        info!("Attestation Response: {:?}", attestation_response);
                        // TODO CS: stream attestation response to client
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
