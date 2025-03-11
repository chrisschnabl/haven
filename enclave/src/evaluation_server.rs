// src/server.rs

use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
use tracing::{error, info, instrument};
use futures::StreamExt;

use crate::typestate::ModelServer;

#[instrument]
pub async fn run_server(port: u32) -> Result<()> {
    // Bind vsock server
    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr).context("Unable to bind Virtio listener")?;
    info!("Listening for connections on port: {}", port);

    let mut incoming = listener.incoming();
    while let Some(stream_result) = incoming.next().await {
        match stream_result {
            Ok(stream) => {
                info!("Got connection from client");
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream).await {
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
async fn handle_client(stream: VsockStream) -> Result<()> {
    info!("Starting new client session");
    
    // Initialize server with typestate pattern
    let server = ModelServer::new(stream);
    
    // Execute the protocol flow using typestate transitions
    let server = server.receive_llama_model().await?;
    info!("LLaMA model received successfully");
    
    let server = server.receive_bert_model().await?;
    info!("BERT model received successfully");
    
    let server = server.receive_evaluation_dataset().await?;
    info!("Evaluation dataset received successfully");
    
    let server = server.run_evaluation().await?;
    info!("Evaluation completed successfully");
    
    let server = server.generate_attestation().await?;
    info!("Attestation generated successfully");
    
    server.complete_session().await?;
    info!("Session completed successfully");
    
    Ok(())
}