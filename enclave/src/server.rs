// src/server.rs
use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
use tracing::{error, info, instrument};
use futures::StreamExt;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use crate::vsock::{read_message, write_message, Operation, Message};
use crate::attestation::generate_attestation;

use llama_runner::{
    LlamaConfig,
    start_llama_thread,
    LlamaActorHandle,
};

#[instrument]
pub async fn run_server(port: u32) -> Result<()> {
    // 1) Start the dedicated llama thread here
    //    Initially, you might not have "model.gguf" yet (since you're about to receive it),
    //    so just create a default config. 
    let config = LlamaConfig::new(); // TODO CS: make config transparent for API
    let (actor_handle, _actor_thread) = start_llama_thread(config);

    // 2) Bind vsock server
    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr).context("Unable to bind Virtio listener")?;
    info!("Listening for connections on port: {}", port);

    let mut incoming = listener.incoming();
    while let Some(stream_result) = incoming.next().await {
        match stream_result {
            Ok(mut stream) => {
                info!("Got connection from client");
                // Pass actor_handle.clone() to each new connection
                let actor_clone = actor_handle.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_incoming_messages(actor_clone, &mut stream).await {
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

#[instrument(skip(stream, actor))]
async fn handle_incoming_messages(
    actor: LlamaActorHandle,
    stream: &mut VsockStream,
) -> Result<()> {
    let mut file = None;
    let mut pb = None;
    let mut total_received = 0u64;

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
                // If not already open, create "model.gguf"
                if file.is_none() {
                    file = Some(
                        File::create("model.gguf")
                            .await
                            .context("Failed to create file 'model.gguf'")?,
                    );
                    pb = Some(ProgressBar::new_spinner());
                    if let Some(ref bar) = pb {
                        bar.set_style(
                            ProgressStyle::default_spinner()
                                .template("{spinner} [{elapsed_precise}] {msg}")
                                .unwrap(),
                        );
                        bar.set_message("Receiving data...");
                        bar.enable_steady_tick(Duration::from_millis(100));
                    }
                }

                // Write incoming chunk to file
                if let Some(ref mut f) = file {
                    f.write_all(&msg.data).await?;
                    total_received += msg.data.len() as u64;
                    if let Some(ref bar) = pb {
                        bar.set_message(format!("Wrote {} bytes to 'model.gguf'", total_received));
                    }
                }
            }

            Operation::EofFile => {
                if let Some(ref bar) = pb {
                    bar.finish_with_message("File transfer complete. Received EOF marker.");
                }
                info!("Received end-of-file marker. Stopping file reception.");

                // We can now load the model if we want, or wait until Operation::Prompt
                // For example, do nothing here, or do:
                // actor.load_model("model.gguf".to_string()).await?;
                // info!("Model loaded after file reception.");
                break;
            }

            Operation::Prompt => {
                // 1) If we haven't loaded the model yet, do it now:
                if file.is_none() {
                    actor.load_model("model.gguf".to_string()).await?;
                    info!("Loaded model in actor thread.");
                }

                // 2) Convert the prompt from bytes
                let prompt = match String::from_utf8(msg.data.clone()) {
                    Ok(p) => p,
                    Err(_) => "Default prompt.".to_string(),
                };
                info!("Received prompt: {}", prompt);

                let (mut token_rx, final_rx) = actor.generate_stream(prompt).await?;
                info!("--- Streaming tokens ---");
                let mut collected = String::new();
                // TODO CS: the stream is not working somehow
                while let Some(token) = token_rx.recv().await {
                    let token_msg = Message {
                        op: Operation::Prompt,
                        data: token.as_bytes().to_vec(),
                    };
                    write_message(stream, &token_msg).await?;
                    collected.push_str(&token);
                }

                // TODO CS: tracing here
                println!("\n--- Done streaming tokens ---");
                
                let msg = Message {
                    op: Operation::EofPrompt,
                    data: vec![],
                };
                write_message(stream, &msg).await?;

                match final_rx.await {
                    Ok(Ok(())) => info!("Generation succeeded."),
                    Ok(Err(e)) => error!("Generation error: {:?}", e),
                    Err(_) => error!("Actor dropped the final result channel."),
                }

                // 4) Perform attestation if needed
                // (this might be blocking, so you can do block_in_place or a separate spawn_blocking)
                match tokio::task::block_in_place(|| {
                    generate_attestation("my-model-id", "input", &collected)
                }) {
                    Ok(attestation_response) => {
                        info!("Attestation Response successfully generated");
                        // TODO: send attestation response to client
                        let msg = Message {
                            op: Operation::Attestation,
                            data: attestation_response,
                        };
                        write_message(stream, &msg).await?;
                    }
                    Err(e) => {
                        error!("Failed to generate attestation: {:?}", e);
                    }
                }

                break;
            }
            _ => {
                error!("Unexpected operation: {:?}", msg.op);
                break;
            }
        }
    }

    // TODO CS: do evaluation here, e.g. run BERT
    Ok(())
}