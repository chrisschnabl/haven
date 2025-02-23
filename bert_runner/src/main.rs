use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use std::io::Write;

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn main() {
    // Set the model path to your BERT embedding GGUF model
    let model_path = "bert-base-uncased-Q8_0.gguf";
    let backend = LlamaBackend::init().expect("failed to init backend");
    let params = LlamaModelParams::default();

    // Use a prompt for which we want to compute embeddings.
    let prompt = "<|im_start|>user\nHello! how are you?<|im_end|>\n<|im_start|>assistant\n".to_string();

    // Load the model
    let model = LlamaModel::load_from_file(&backend, model_path, &params)
        .expect("unable to load model");

    // Create a context with embeddings enabled
    let ctx_params = LlamaContextParams::default().with_embeddings(true);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("unable to create the llama_context");

    // Tokenize the prompt with AddBos to ensure proper tokenization
    let tokens_list = model
        .str_to_token(&prompt, AddBos::Always)
        .unwrap_or_else(|_| panic!("failed to tokenize {prompt}"));

    let mut batch = LlamaBatch::new(512, 1);

    let last_index = tokens_list.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)
            .expect("failed to add token to batch");
    }
    // Decode the batch to compute embeddings
    ctx.decode(&mut batch).expect("llama_decode() failed");

    // Instead of generating text, extract the embedding vector for the sequence (sequence 0)
    let embedding = ctx
        .embeddings_seq_ith(0)
        .expect("failed to get embeddings for sequence");

    // Print the resulting embedding vector
    println!("Embeddings:\n{:?}", embedding);

    // Optionally, print timing information
    println!("{}", ctx.timings());
}