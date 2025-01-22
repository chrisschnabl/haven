use anyhow::Result;
use llama_runner::{
    LlamaConfig, 
    start_llama_thread
};

#[tokio::main]
async fn main() -> Result<()> {

    tracing_subscriber::fmt()
    .init();

    let config = LlamaConfig::new("llama-2-7b.Q4_K_M.gguf");
    let (actor_handle, _thread_join_handle) = start_llama_thread(config);

    actor_handle.load_model("llama-2-7b.Q4_K_M.gguf".into()).await?;
    println!("Model loaded successfully!\n");

    let prompt = "What is the meaning of life?".to_string();
    let full_text = actor_handle.generate(prompt).await?;
    println!("---- Full output ----\n{full_text}\n");

    let stream_prompt = "Please write a short poem:".to_string();
    let (mut token_rx, final_rx) = actor_handle.generate_stream(stream_prompt).await?;
    println!("---- Streaming tokens ----");
    while let Some(token) = token_rx.recv().await {
        print!("{token}");
    }
    println!("\n---- End of stream ----");

    // Check final status from the actor
    match final_rx.await {
        Ok(Ok(())) => println!("Generation completed successfully."),
        Ok(Err(e)) => eprintln!("Generation error: {e:?}"),
        Err(_)     => eprintln!("Actor dropped final result channel."),
    }

    Ok(())
}
