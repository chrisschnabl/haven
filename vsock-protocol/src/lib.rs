pub mod io;
pub mod handler;

pub use io::{read_message, write_message};
pub use handler::{run_server, make_handler, ServerHandler};

use anyhow::Result;
use tokio_vsock::VsockStream;
use std::future::Future;
use handler::connect_to_server;

pub async fn run_client_with<F, Fut>(
    cid: u32,
    port: u32,
    client_impl: F,
) -> Result<()>
where
    F: FnOnce(VsockStream) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let stream = connect_to_server(cid, port).await?;
    client_impl(stream).await
}

pub async fn run_server_with<F, Fut>(
    port: u32,e
    server_impl: F,
) -> Result<()>
where
    F: Fn(VsockStream) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    run_server(port, make_handler(server_impl)).await
}