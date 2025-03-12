mod messages;
mod file_transfer;
mod client_state;
mod server_state;

use anyhow::Result;
use structopt::StructOpt;
use tracing::{info, instrument, Level};
use tracing_subscriber::FmtSubscriber;

use tokio_vsock::VsockStream;
use vsock::{run_server_with, run_client_with};
use crate::server_state::ModelServer;
use crate::client_state::ModelClient;

#[derive(Debug, StructOpt)]
#[structopt(name = "model-evaluator", about = "Secure Model Evaluation System")]
enum Opt {
    #[structopt(name = "server")]
    Server {
        #[structopt(short, long, default_value = "5000")]
        port: u32,
    },
    
    #[structopt(name = "client")]
    Client {
        #[structopt(short, long)]
        cid: u32,
        
        #[structopt(short, long, default_value = "5000")]
        port: u32,
        
        #[structopt(long)]
        llama_model: String,
        
        #[structopt(long)]
        bert_model: String,
        
        #[structopt(long)]
        dataset: String,
    },
}


#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
    
    let opt = Opt::from_args();
    
    match opt {
        Opt::Server { port } => {
            info!("Starting server on port {}", port);
            run_server(port).await?;
        }
        
        Opt::Client { cid, port, llama_model, bert_model, dataset } => {
            info!("Starting client, connecting to CID {}, port {}", cid, port);
            run_client(cid, port, &llama_model, &bert_model, &dataset).await?;
        }
    }
    
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


#[instrument]
pub async fn run_client(
    cid: u32, 
    port: u32, 
    llama_path: &str, 
    bert_path: &str, 
    dataset_path: &str
) -> Result<()> {
    info!("Starting client for model evaluation workflow");
    
    run_client_with(cid, port, |stream| async move {
        handle_server(stream, llama_path, bert_path, dataset_path).await
    }).await
}

#[instrument]
async fn handle_server(stream: VsockStream, llama_path: &str, bert_path: &str, dataset_path: &str) -> Result<()> {

    let client = ModelClient::new();

    let client = client.connect(stream);
    let client = client.send_llama_model(llama_path).await?;
    let client = client.send_bert_model(bert_path).await?;
    let client = client.send_dataset(dataset_path).await?;
    let client = client.wait_for_evaluation().await?;
    let client = client.verify_attestation().await?;
    client.complete_session().await?;
    
    Ok(())
}