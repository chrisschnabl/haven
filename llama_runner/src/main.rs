use llama_runner::{start_llama_thread, LlamaConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1) Create a config
    let config = LlamaConfig::new("llama-2-7b.Q4_K_M.gguf");

    // 2) Start the dedicated thread
    let (actor_handle, _thread_handle) = start_llama_thread(config);

    // 3) Load a model (optional if already loaded)
    actor_handle.load_model("llama-2-7b.Q4_K_M.gguf".to_string()).await?;

    // 4) Generate
    let text = actor_handle.generate("Hello llama!".to_string()).await?;
    println!("Final generation:\n{text}");

    // 5) Or do streaming
    let mut token_stream = actor_handle.generate_stream("Say a joke:".to_string()).await?;
    while let Some(token) = token_stream.recv().await {
        print!("{token}");
    }
    println!("\nDone streaming.");

    Ok(())
}


// TODO CS: make this tokio compatible
// TOOD CS: do not take this from a file but already from memory  