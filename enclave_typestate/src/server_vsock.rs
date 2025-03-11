use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
use tracing::{info, error, instrument};
use anyhow::{Context, Result};
use crate::server_state::ModelServer;
use futures::StreamExt;

#[instrument]
pub async fn run_server(port: u32) -> Result<()> {
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
    let server = ModelServer::new(stream);
    let server = server.receive_llama_model().await?;    
    let server = server.receive_bert_model().await?;
    let server = server.receive_dataset().await?;
    let server = server.run_evaluation().await?;
    let server = server.generate_attestation().await?;
    server.complete_session().await?;
    
    Ok(())
}