use anyhow::{Result};
use tracing::{info, instrument};
use crate::client_state::ModelClient;

#[instrument]
pub async fn run_client(
    cid: u32, 
    port: u32, 
    llama_path: &str, 
    bert_path: &str, 
    dataset_path: &str
) -> Result<()> {
    info!("Starting client for model evaluation workflow");
    
    let client = ModelClient::new();
    
    let client = client.connect(cid, port).await?;
    let client = client.send_llama_model(llama_path).await?;
    let client = client.send_bert_model(bert_path).await?;
    let client = client.send_dataset(dataset_path).await?;
    let client = client.wait_for_evaluation().await?;
    let client = client.verify_attestation().await?;
    
    client.complete_session().await?;
    
    Ok(())
}