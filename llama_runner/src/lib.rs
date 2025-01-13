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

/// Simple configuration struct for our model usage.
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
pub struct LlamaRunner {
    config: LlamaConfig,

    backend: Option<LlamaBackend>,
    model: Option<&'static LlamaModel>,
    context: Option<llama_cpp_2::context::LlamaContext<'static>>,
    sampler: Option<LlamaSampler>,
}

impl LlamaRunner {
    /// Create the runner with config. We haven't loaded the model yet.
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
        // 1. Init backend
        let backend = LlamaBackend::init().context("Failed to initialize LlamaBackend")?;

        // 2. Load model
        let model_params = LlamaModelParams::default();
        println!("Loading model from path: {:?}", self.config.model_path);
        let model_box = Box::new(
            LlamaModel::load_from_file(&backend, &self.config.model_path, &model_params)
                .context("Unable to load model")?,
        );
        // Box::leak -> &'static LlamaModel
        let model: &'static LlamaModel = Box::leak(model_box);
        println!("Model loaded successfully!");

        // 3. Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(self.config.context_size))
            .with_n_threads(self.config.threads);

        let context = model
            .new_context(&backend, ctx_params)
            .context("Unable to create the llama_context")?;

        // 4. Create sampler
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::dist(self.config.seed as u32),
            LlamaSampler::greedy(),
        ]);

        // Store all in self
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
        // on_token: gets each new token as soon as it's available
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

        // 1. Tokenize the prompt
        let tokens_list = model
            .str_to_token(prompt, AddBos::Always)
            .context(format!("Failed to tokenize '{prompt}'"))?;

        // 2. Check context sizes
        let n_ctx = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);
        if n_kv_req > n_ctx {
            bail!("n_kv_req > n_ctx: required KV cache size is too big.");
        }
        if tokens_list.len() >= n_len.try_into()? {
            bail!("The prompt is too long; it has more tokens than n_len.");
        }

        // 3. (Optional) output the prompt tokens first
        for &token in &tokens_list {
            let token_str = model.token_to_str(token, Special::Tokenize)?;
            // Send to callback
            on_token(&token_str);
            // Also print to stdout or collect in `output`
            print!("{}", token_str);
            output.push_str(&token_str);
        }
        std::io::stdout().flush()?;

        // 4. Create a batch
        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        // 5. Initial decode
        ctx.decode(&mut batch).context("llama_decode() failed")?;

        // 6. Main loop for generation
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let t_main_start = ggml_time_us();

        while n_cur <= n_len {
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // If it's an end-of-generation token, break
            if model.is_eog_token(token) {
                println!();
                break;
            }

            let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
            let output_str = String::from_utf8(output_bytes)?;
            // Immediately call callback with the new token
            on_token(&output_str);

            // Also print or accumulate
            print!("{}", output_str);
            std::io::stdout().flush()?;
            output.push_str(&output_str);

            // Prepare next iteration
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            // Evaluate again
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

        // (Optional) print context timings
        println!("{}", ctx.timings());

        Ok(output)
    }

    /// (Optional) If you want a blocking version that just returns the final string
    /// (just calls `generate_stream` internally).
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        self.generate_stream(prompt, |_| {})
    }
}
