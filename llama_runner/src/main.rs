use anyhow::Result;
use llama_runner::{LlamaConfig, start_llama_thread};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .init();

    let config = LlamaConfig::new();
    let (actor, _jh) = start_llama_thread(config);
    actor.load_model("llama-2-7b.Q4_K_M.gguf".to_string()).await?;

    let out = actor.generate("Hello llama!".to_string()).await?;
    println!("Full generation: {out}");

    Ok(())
}