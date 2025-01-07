use std::io::{self, Write};
use llama_cpp::{LlamaModel, LlamaParams, SessionParams, TokensToStrings};
use llama_cpp::standard_sampler::StandardSampler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the model
    let model = LlamaModel::load_from_file("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", LlamaParams::default())?;

    // Create a session
    let mut ctx = model.create_session(SessionParams::default())?;

    // Advance the context with a prompt
    ctx.advance_context("This is the story of a man named Stanley.")?;

    // Start completing with the standard sampler
    let completion_handle = ctx.start_completing_with(StandardSampler::default(), 1024)?;

    // Convert tokens to strings
    let mut tokens_to_strings = TokensToStrings::new(completion_handle, model);

    // Iterate over the generated tokens
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