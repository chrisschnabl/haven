use anyhow::Result;
use llama_runner::{LlamaConfig, LlamaRunner};

fn main() -> Result<()> {
    let config = LlamaConfig::new("/home/ec2-user/haven/vsock/async/llama-2-7b.Q2_K.gguf");
    let mut runner = LlamaRunner::new(config);
    runner.load_model()?;

    println!("---- Streaming tokens for prompt #1 ----");
    let prompt = "Hello, this is a test prompt. Tell me a joke:";
    let text = runner.generate_stream(prompt, |_token| {
    })?;
    println!("\n{text}\n");

    Ok(())
}
