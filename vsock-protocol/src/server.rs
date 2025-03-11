use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
use tracing::{error, info, instrument};
use futures::StreamExt;
use std::future::Future;
use std::pin::Pin;

/// Type alias for a server handler function
pub type ServerHandler = Box<dyn Fn(VsockStream) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;

#[instrument]
pub async fn run_server(port: u32, handler: ServerHandler) -> Result<()> {
    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(&addr).context("Unable to bind Virtio listener")?;
    info!("Listening for connections on port: {}", port);

    let mut incoming = listener.incoming();
    while let Some(stream_result) = incoming.next().await {
        match stream_result {
            Ok(stream) => {
                info!("Got connection from client");
                tokio::spawn(async move {
                    if let Err(e) = handler(stream).await {
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

/// A helper function to create a server handler from a handler function
pub fn make_handler<F, Fut>(f: F) -> ServerHandler 
where
    F: Fn(VsockStream) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    Box::new(move |stream| Box::pin(f(stream)))
}