use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::convert::TryInto;
use std::io::Write;
use std::num::NonZeroU32;
use std::time::Duration;

// TODO CS: allow for easier config and more parameters
pub struct LlamaConfig {
    pub model_path: String,
    pub seed: i64,
    pub threads: i32,
    pub context_size: NonZeroU32,
    pub n_len: i32,
}

impl LlamaConfig {
    pub fn new(model_path: &str) -> Self {
        Self {
            model_path: model_path.into(),
            seed: 1337,
            threads: 2,
            context_size: NonZeroU32::new(2048).unwrap(),
            n_len: 32,
        }
    }
}

/// A runner that separates `load_model()` from generation.
/// Uses `'static` references to keep it simple (via `Box::leak`).
// TODO CS: rethink this abscration to be thread-safe and nicer to use 
pub struct LlamaRunner {
    config: LlamaConfig,

    backend: Option<LlamaBackend>,
    model: Option<&'static LlamaModel>,
    context: Option<llama_cpp_2::context::LlamaContext<'static>>,
    sampler: Option<LlamaSampler>,
}

impl LlamaRunner {
    pub fn new(config: LlamaConfig) -> Self {
        Self {
            config,
            backend: None,
            model: None,
            context: None,
            sampler: None,
        }
    }

    /// Load the model and create a context/sampler. Call this once before generating.
    pub fn load_model(&mut self) -> Result<()> {
        let backend = LlamaBackend::init().context("Failed to initialize LlamaBackend")?;

        let model_params = LlamaModelParams::default();
        println!("Loading model from path: {:?}", self.config.model_path);
        let model_box = Box::new(
            LlamaModel::load_from_file(&backend, &self.config.model_path, &model_params)
                .context("Unable to load model")?,
        );
        // Box::leak -> &'static LlamaModel
        let model: &'static LlamaModel = Box::leak(model_box);
        println!("Model loaded successfully!");

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(self.config.context_size))
            .with_n_threads(self.config.threads);

        let context = model
            .new_context(&backend, ctx_params)
            .context("Unable to create the llama_context")?;

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::dist(self.config.seed as u32),
            LlamaSampler::greedy(),
        ]);

        self.backend = Some(backend);
        self.model = Some(model);
        self.context = Some(context);
        self.sampler = Some(sampler);

        Ok(())
    }

    /// Synchronous "streaming" generation: call `on_token` for each new token.
    ///
    /// If you want to capture the entire output, you can still accumulate tokens in your callback,
    /// or look at the return value which is the final text.
    pub fn generate_stream<F>(&mut self, prompt: &str, mut on_token: F) -> Result<String>
    where
        F: FnMut(&str),
    {
        let model = self
            .model
            .as_ref()
            .expect("Model not loaded. Call load_model() first!");
        let ctx = self
            .context
            .as_mut()
            .expect("Context not loaded. Call load_model() first!");
        let sampler = self
            .sampler
            .as_mut()
            .expect("Sampler not loaded. Call load_model() first!");

        let mut output = String::new();
        let n_len = self.config.n_len;

        let tokens_list = model
            .str_to_token(prompt, AddBos::Always)
            .context(format!("Failed to tokenize '{prompt}'"))?;

        let n_ctx = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);
        if n_kv_req > n_ctx {
            bail!("n_kv_req > n_ctx: required KV cache size is too big.");
        }
        if tokens_list.len() >= n_len.try_into()? {
            bail!("The prompt is too long; it has more tokens than n_len.");
        }

        for &token in &tokens_list {
            let token_str = model.token_to_str(token, Special::Tokenize)?;
            on_token(&token_str); // TODO CS: transform this into a tokio stream
            print!("{}", token_str);
            output.push_str(&token_str);
        }
        std::io::stdout().flush()?;

        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        // 5. Initial decode
        ctx.decode(&mut batch).context("llama_decode() failed")?;

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let t_main_start = ggml_time_us();

        while n_cur <= n_len {
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if model.is_eog_token(token) {
                println!();
                break;
            }

            let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
            let output_str = String::from_utf8(output_bytes)?;
            on_token(&output_str);

            print!("{}", output_str);
            std::io::stdout().flush()?;
            output.push_str(&output_str);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch).context("Failed to eval")?;
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

        // TODO CS: move to tracing
        println!("{}", ctx.timings());

        Ok(output)
    }
}