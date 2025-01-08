use std::io::{self, Write};
use std::env;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams, TokensToStrings};
use llama_cpp::standard_sampler::StandardSampler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match env::current_dir() {
        Ok(path) => println!("Current directory: {}", path.display()),
        Err(e) => eprintln!("Error getting current directory: {}", e),
    }

    let llama_params = LlamaParams {
        n_threads: 1, 
        ..LlamaParams::default()       
    };

    let model = LlamaModel::load_from_file("llama-2-7b.Q2_K.gguf", llama_params)?;
    println!("Model loaded successfully!");

    let mut ctx = model.create_session(SessionParams::default())?;
    println!("Model contexxt created.");

    ctx.advance_context("This is the story of a man named Stanley.")?;
    println!("Model context advanced.");

    // Start completing with the standard sampler
    let completion_handle = ctx.start_completing_with(StandardSampler::default(), 1024)?;

    let mut tokens_to_strings = TokensToStrings::new(completion_handle, model);

    let max_tokens = 1024;
    let mut decoded_tokens = 0;
    while let Some(token) = tokens_to_strings.next() {
        print!("{}", token);
        io::stdout().flush()?;
        decoded_tokens += 1;
        if decoded_tokens >= max_tokens {
            break;
        }   
    }

    Ok(())
}