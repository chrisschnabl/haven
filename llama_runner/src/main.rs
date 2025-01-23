use anyhow::Result;
use llama_runner::{LlamaConfig, start_llama_thread};

// Example how to use the actor handle
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .init();

    let config = LlamaConfig::new();
    let (actor, _jh) = start_llama_thread(config);
    actor.load_model("llama-2-7b.Q4_K_M.gguf".to_string()).await?;
    
    
    let (mut token_rx, reply_rx) = actor.generate_stream("Hello llama!".to_string()).await?;
    let mut out = String::new();
    while let Some(token) = token_rx.recv().await {
        out.push_str(&token);
        print!("{token}");
    }
    reply_rx.await??;
    println!("Full generation: {out}");

    Ok(())
}