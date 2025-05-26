use anyhow::Result;
use llama_runner::{LlamaConfig, start_llama_thread};
use tracing::{info, instrument};

// Example on how to use the actor handle
#[instrument]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .init();

    info!("Starting Llama runner");
    let config = LlamaConfig::new();
    let (actor, _jh) = start_llama_thread(config);
    
    info!("Loading model");
    actor.load_model("llama-2-7b.Q4_K_M.gguf".to_string()).await?;
    
    info!("Starting text generation");
    let (mut token_rx, reply_rx) = actor.generate_stream("Hello llama!".to_string()).await?;
    let mut out = String::new();
    while let Some(token) = token_rx.recv().await {
        out.push_str(&token);
        print!("{token}");
    }
    reply_rx.await??;
    info!("Generation complete");
    println!("Full generation: {out}");

    Ok(())
}