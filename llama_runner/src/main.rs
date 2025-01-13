use anyhow::Result;
use llama_runner::{LlamaConfig, LlamaRunner};

fn main() -> Result<()> {
    // 1. Create config
    let config = LlamaConfig::new("/home/ec2-user/haven/vsock/async/llama-2-7b.Q2_K.gguf");

    // 2. Create runner (but we haven't loaded the model yet)
    let mut runner = LlamaRunner::new(config);

    // 3. Load model and context
    runner.load_model()?;

    // 4a. Example: streaming tokens with a callback
    println!("---- Streaming tokens for prompt #1 ----");
    let prompt = "Hello, this is a test prompt. Tell me a joke:";
    let final_text_1 = runner.generate_stream(prompt, |token| {
        // Here we just print again, but you could do something else,
        // like sending tokens over a channel or a websocket.
        // For example: ws.send(token).await? if you had an async context
    })?;
    println!("Final text #1:\n{final_text_1}");

    // 4b. Another prompt with streaming
    println!("\n---- Streaming tokens for prompt #2 ----");
    let prompt2 = "Do it again in German:";
    let final_text_2 = runner.generate_stream(prompt2, |token| {
        // Show that we could do something custom with the tokens
        // Here, for example, we track how many tokens we got.
        // But let's keep it simple and just do nothing special.
    })?;
    println!("Final text #2:\n{final_text_2}");

    Ok(())
}
