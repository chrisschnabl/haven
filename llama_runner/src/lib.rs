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
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tracing::{error, info};

/// Configuration for the runner
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

/// A blocking Llama runner that is **not Send**.
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

    pub fn load_model(&mut self) -> Result<()> {
        info!("Loading model from path: {}", self.config.model_path);

        let backend = LlamaBackend::init().context("Failed to initialize LlamaBackend")?;
        let model_params = LlamaModelParams::default();

        let model_box = Box::new(
            LlamaModel::load_from_file(&backend, &self.config.model_path, &model_params)
                .context("Unable to load model")?,
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

    /// Blocking generation. Calls `on_token` for each new token.
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

        // Prepare a batch
        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        // Initial decode
        ctx.decode(&mut batch).context("Failed to decode prompt tokens")?;

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let t_main_start = ggml_time_us();

        while n_cur <= n_len {
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // End if EOG
            if model.is_eog_token(token) {
                break;
            }

            let output_bytes = model
                .token_to_bytes(token, Special::Tokenize)
                .context("Failed to convert token to bytes")?;
            let output_str = String::from_utf8(output_bytes)
                .context("Invalid UTF-8 from token")?;

            on_token(&output_str);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch).context("Failed to decode next token")?;
            n_decode += 1;
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

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

/// Commands the dedicated thread can process
pub enum LlamaCommand {
    LoadModel {
        model_path: String,
        reply: oneshot::Sender<Result<()>>,
    },
    /// Always generate tokens, sending them to `token_tx`. 
    /// After finishing or error, send the final Result via `reply`.
    Generate {
        prompt: String,
        token_tx: tokio_mpsc::Sender<String>,
        reply: oneshot::Sender<Result<()>>,
    },
}

/// A handle you can clone to send commands from async code.
#[derive(Clone)]
pub struct LlamaActorHandle {
    cmd_tx: Sender<LlamaCommand>,
}

impl LlamaActorHandle {
    pub fn new(cmd_tx: Sender<LlamaCommand>) -> Self {
        Self { cmd_tx }
    }

    /// Load a model, awaiting the final result
    pub async fn load_model(&self, model_path: String) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = LlamaCommand::LoadModel {
            model_path,
            reply: reply_tx,
        };

        // Send the command via std::sync::mpsc (blocking), 
        // because we're *outside* the dedicated thread. That’s fine.
        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread is closed"))?;

        // Now wait for the final result in an async oneshot
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Actor dropped the reply"))?
    }

    /// Generate the entire output. Collect tokens from the channel into one String.
    pub async fn generate(&self, prompt: String) -> Result<String> {
        let (token_tx, mut token_rx) = tokio_mpsc::channel(64);
        let (reply_tx, reply_rx) = oneshot::channel();

        let cmd = LlamaCommand::Generate {
            prompt,
            token_tx,
            reply: reply_tx,
        };

        // Send the command
        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread is closed"))?;

        // Collect tokens as they arrive
        let mut output = String::new();
        while let Some(token) = token_rx.recv().await {
            output.push_str(&token);
        }

        // Then check final success/failure
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("Actor dropped the reply"))??;

        Ok(output)
    }

    /// Generate as a stream: return (token receiver, final status oneshot)
    pub async fn generate_stream(
        &self,
        prompt: String,
    ) -> Result<(tokio_mpsc::Receiver<String>, oneshot::Receiver<Result<()>>)> {
        let (token_tx, token_rx) = tokio_mpsc::channel(64);
        let (reply_tx, reply_rx) = oneshot::channel();

        let cmd = LlamaCommand::Generate {
            prompt,
            token_tx,
            reply: reply_tx,
        };

        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread is closed"))?;

        Ok((token_rx, reply_rx))
    }
}

/// Spawn a dedicated OS thread that owns `LlamaRunner` and processes commands in a loop (blocking).
pub fn start_llama_thread(config: LlamaConfig) -> (LlamaActorHandle, thread::JoinHandle<()>) {
    // std::sync::mpsc for commands
    let (cmd_tx, cmd_rx) = mpsc::channel::<LlamaCommand>();

    let handle = LlamaActorHandle::new(cmd_tx);

    // Spawn the dedicated thread
    let join_handle = thread::spawn(move || {
        // Create & own the runner in this thread
        let mut runner = LlamaRunner::new(config);

        // Process commands in a blocking loop
        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                LlamaCommand::LoadModel { model_path, reply } => {
                    runner.config.model_path = model_path;
                    let res = runner.load_model();
                    if let Err(e) = &res {
                        error!("Failed to load model: {:?}", e);
                    }
                    let _ = reply.send(res);
                }
                LlamaCommand::Generate {
                    prompt,
                    token_tx,
                    reply,
                } => {
                    let res = runner.generate_blocking(&prompt, |token_str| {
                        // push tokens to the async channel
                        let _ = token_tx.blocking_send(token_str.to_owned());
                    });
                    // Dropping token_tx => no more tokens
                    drop(token_tx);

                    if let Err(e) = &res {
                        error!("Generate error: {:?}", e);
                    }
                    let _ = reply.send(res);
                }
            }
        }

        info!("Actor thread: command channel closed, exiting.");
    });

    (handle, join_handle)
}