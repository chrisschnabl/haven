use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::convert::TryInto;
use tracing::info;

use crate::config::LlamaConfig;

/// A blocking Llama runner that is **not `Send`**.
pub struct LlamaRunner {
    pub config: LlamaConfig,
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

    /// Load the model from `self.config.model_path`.
    pub fn load_model(&mut self) -> Result<()> {
        // Start of Selection
        let model_path = self.config.model_path.as_ref().context("Model path is not set")?;
        
        info!("Loading model from path: {}", model_path);
        let backend = LlamaBackend::init().context("Failed to initialize LlamaBackend")?;

        let model_params = LlamaModelParams::default();
        let model_box = Box::new(
            LlamaModel::load_from_file(&backend, model_path, &model_params)
                .context("Unable to load model")?
        );

        let model: &'static LlamaModel = Box::leak(model_box);

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

        info!("Model loaded successfully!");
        Ok(())
    }

    /// Blocking generation that calls `on_token` for each token produced.
    pub fn generate_blocking<F>(&mut self, prompt: &str, mut on_token: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => bail!("Model not loaded; call load_model() first."),
        };
        let ctx = match self.context.as_mut() {
            Some(c) => c,
            None => bail!("Context not loaded; call load_model() first."),
        };
        let sampler = match self.sampler.as_mut() {
            Some(s) => s,
            None => bail!("Sampler not loaded; call load_model() first."),
        };

        let n_len = self.config.n_len;
        let tokens_list = model
            .str_to_token(prompt, AddBos::Always)
            .context(format!("Failed to tokenize prompt: {prompt}"))?;

        let n_ctx = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);
        if n_kv_req > n_ctx {
            bail!("n_kv_req > n_ctx (required KV cache size is too big)");
        }
        if tokens_list.len() >= n_len.try_into()? {
            bail!("Prompt is too long; more tokens than n_len.");
        }

        // Output the prompt tokens
        for &token in &tokens_list {
            let token_str = model.token_to_str(token, Special::Tokenize)?;
            on_token(&token_str);
        }

        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        // decode initial prompt
        ctx.decode(&mut batch).context("Failed to decode prompt tokens")?;

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let t_main_start = ggml_time_us();

        while n_cur <= n_len {
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // end if EOG
            if model.is_eog_token(token) {
                break;
            }

            let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
            let output_str = String::from_utf8(output_bytes)
                .context("Failed to convert token to utf8")?;
            on_token(&output_str);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .context("Failed to decode next token")?;
            n_decode += 1;
        }

        let t_main_end = ggml_time_us();
        let duration = std::time::Duration::from_micros((t_main_end - t_main_start) as u64);

        info!(
            "Decoded {} tokens in {:.2}s, speed {:.2} t/s\n{}",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32(),
            ctx.timings()
        );

        Ok(())
    }
}