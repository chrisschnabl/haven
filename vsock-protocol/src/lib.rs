
pub mod protocol;
pub mod server;
pub mod client;
pub mod file_transfer;

pub use protocol::{Message, Operation, read_message, write_message};
pub use server::{run_server, make_handler, ServerHandler};
pub use file_transfer::{connect_to_server, receive_file, BUFFER_SIZE};

/// Takes a client implementation function and runs it with the given parameters
pub async fn run_client_with<F>(
    cid: u32,
    port: u32,
    client_impl: F,
) -> anyhow::Result<()>
where
    F: FnOnce(tokio_vsock::VsockStream) -> anyhow::Result<()>,
{
    let stream = connect_to_server(cid, port).await?;
    client_impl(stream)
}

/// That takes a server implementation function and runs it with the given parameters
pub async fn run_server_with<F>(
    port: u32,
    server_impl: F,
) -> anyhow::Result<()>
where
    F: Fn(tokio_vsock::VsockStream) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>> + Send + Sync + 'static,
{
    run_server(port, make_handler(server_impl)).await
}