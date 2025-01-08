use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::time::Duration;
use std::convert::TryInto;
use std::io::Write;

fn main() -> Result<()> {
    // Params
    let model_path = "/home/ec2-user/haven/llama-rs/llama-2-7b.Q2_K.gguf";
    let prompt = "Hello, this is a test prompt.";
    let n_len = 32;
    let seed = 1337;
    let threads = 2;
    let ctx_size = NonZeroU32::new(2048).unwrap();


    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();

    println!("Loading model from path: {:?}", model_path);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "Unable to load model")?;
    println!("Model loaded successfully!");

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_threads(threads);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "Unable to create the llama_context")?;

    // Tokenize the prompt
    let tokens_list = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("Failed to tokenize {prompt}"))?;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);
    println!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");
    if n_kv_req > n_cxt {
        bail!(
            "n_kv_req > n_ctx, the required kv cache size is not big enough. \
            Either reduce n_len or increase n_ctx"
        )
    }
    if tokens_list.len() >= n_len.try_into()? {
        bail!("The prompt is too long, it has more tokens than n_len")
    }

    // Print the prompt token-by-token
    for token in &tokens_list {
        print!("{}", model.token_to_str(*token, Special::Tokenize)?);
    }
    std::io::stdout().flush()?;

    // Create a batch for decoding
    let mut batch = LlamaBatch::new(512, 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // Main loop for token generation
    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;
    let t_main_start = ggml_time_us();
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(seed),
        LlamaSampler::greedy(),
    ]);
    while n_cur <= n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            println!();
            break;
        }

        let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let output_string = String::from_utf8(output_bytes)?;
        print!("{output_string}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;

        n_cur += 1;

        ctx.decode(&mut batch)
            .with_context(|| "Failed to eval")?;

        n_decode += 1;
    }

    println!();
    let t_main_end = ggml_time_us();

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

    println!(
        "Decoded {} tokens in {:.2} s, speed {:.2} t/s",
        n_decode,
        duration.as_secs_f32(),
        n_decode as f32 / duration.as_secs_f32()
    );

    println!("{}", ctx.timings());

    Ok(())
}