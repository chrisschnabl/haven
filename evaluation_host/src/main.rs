mod host;

use anyhow::Result;
use structopt::StructOpt;
use tracing::{info, instrument, Level};
use tracing_subscriber::FmtSubscriber;

use tokio_vsock::VsockStream;
use vsock::run_client_with;
use host::ModelClient;

#[derive(Debug, StructOpt)]
#[structopt(name = "model-evaluator", about = "Secure Model Evaluation System")]
struct Opt {
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
    
    let opt = Opt::from_args();
    
    info!("Starting client, connecting to CID {}, port {}", opt.cid, opt.port);
    run_client(opt.cid, opt.port, &opt.llama_model, &opt.bert_model, &opt.dataset).await?;
    
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