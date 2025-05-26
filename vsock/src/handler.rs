use anyhow::{Context, Result};
use tokio_vsock::{VsockListener, VsockAddr, VsockStream};
use tracing::{error, info, instrument};
use futures::StreamExt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type ServerHandler = Arc<dyn Fn(VsockStream) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;

#[instrument(skip(handler), fields(port = port))]
pub async fn run_server(port: u32, handler: ServerHandler) -> Result<()> {
    info!("Listening for connections on port: {}", port);
    
    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr)
        .context("Unable to bind Virtio listener")?;

    let mut incoming = listener.incoming();
    while let Some(stream_result) = incoming.next().await {
        match stream_result {
            Ok(stream) => {
                info!("Got connection from client");
                let handler_clone = handler.clone();
                tokio::spawn(async move {
                    if let Err(e) = handler_clone(stream).await {
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

/// Create a server handler from a handler function
pub fn make_handler<F, Fut>(f: F) -> ServerHandler 
where
    F: Fn(VsockStream) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    Arc::new(move |stream| Box::pin(f(stream)))
}


#[instrument]
pub async fn connect_to_server(cid: u32, port: u32) -> Result<VsockStream> {
    info!("Connecting to server at CID {}, port {}", cid, port);
    
    let addr = VsockAddr::new(cid, port);
    let stream = VsockStream::connect(addr)
        .await
        .context("Failed to connect to server")?;
    
    info!("Successfully connected to server");
    Ok(stream)
}