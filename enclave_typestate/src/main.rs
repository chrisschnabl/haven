mod messages;
mod file_transfer;
// TODO CS: better naming and organization
mod client_state;
mod server_state;
mod client_vsock;
mod server_vsock;

use anyhow::Result;
use structopt::StructOpt;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

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
            server_vsock::run_server(port).await?;
        }
        
        Opt::Client { cid, port, llama_model, bert_model, dataset } => {
            info!("Starting client, connecting to CID {}, port {}", cid, port);
            client_vsock::run_client(cid, port, &llama_model, &bert_model, &dataset).await?;
        }
    }
    
    Ok(())
}