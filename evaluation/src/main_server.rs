mod messages;
mod file_transfer;
mod server;

use anyhow::Result;
use structopt::StructOpt;
use tracing::{info, instrument, Level};
use tracing_subscriber::FmtSubscriber;

use vsock::run_server_with;
use crate::server::ModelServer;

#[derive(Debug, StructOpt)]
#[structopt(name = "model-evaluator", about = "Secure Model Evaluation System")]
struct Opt {
    #[structopt(short, long, default_value = "5000")]
    port: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
    
    let opt = Opt::from_args();
    
    info!("Starting server on port {}", opt.port);
    run_server(opt.port).await?;
    
    Ok(())
}

#[instrument]
async fn run_server(port: u32) -> Result<()> {
    info!("Starting server on port {}", port);
    run_server_with(port, |stream| async move {
        handle_client(stream).await
    }).await
}

#[instrument(skip_all)]
async fn handle_client(stream: tokio_vsock::VsockStream) -> Result<()> {
    info!("Handling new client connection");
    
    let server = ModelServer::new(stream);
    let server = server.receive_llama_model().await?;    
    let server = server.receive_bert_model().await?;
    let server = server.receive_dataset().await?;
    let server = server.run_evaluation().await?;
    let server = server.generate_attestation().await?;
    server.complete_session().await?;
    
    info!("Client session completed successfully");
    Ok(())
}