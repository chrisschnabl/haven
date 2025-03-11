// src/main.rs

mod vsock;
mod typestate;
mod evaluation_server;
mod evaluation_client;

use anyhow::Result;
use structopt::StructOpt;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Debug, StructOpt)]
#[structopt(name = "model-evaluator", about = "Secure Model Evaluation System")]
enum Opt {
    /// Run in server mode
    #[structopt(name = "server")]
    Server {
        /// Port to listen on
        #[structopt(short, long, default_value = "5000")]
        port: u32,
    },
    
    /// Run in client mode
    #[structopt(name = "client")]
    Client {
        /// CID of the server to connect to
        #[structopt(short, long)]
        cid: u32,
        
        /// Port to connect to
        #[structopt(short, long, default_value = "5000")]
        port: u32,
        
        /// Path to the LLaMA model file
        #[structopt(long)]
        llama_model: String,
        
        /// Path to the BERT model file
        #[structopt(long)]
        bert_model: String,
        
        /// Path to the evaluation dataset file
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
            evaluation_server::run_server(port).await?;
        }
        
        Opt::Client { cid, port, llama_model, bert_model, dataset } => {
            info!("Starting client, connecting to CID {}, port {}", cid, port);
            evaluation_client::run_client(cid, port, &llama_model, &bert_model, &dataset).await?;
        }
    }
    
    Ok(())
}