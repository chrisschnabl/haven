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
use std::sync::mpsc::{self, Sender, Receiver};
use std::thread;
use std::time::Duration;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

/// Configuration for your LlamaRunner
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

/// A simple blocking Llama runner
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

    /// **Blocking** generation loop; calls a callback on each new token.
    pub fn generate_blocking<F>(&mut self, prompt: &str, mut on_token: F) -> Result<String>
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

        // Print/stream the prompt tokens first
        for &token in &tokens_list {
            let token_str = model.token_to_str(token, Special::Tokenize)?;
            on_token(&token_str);
            output.push_str(&token_str);
        }
        std::io::stdout().flush()?;

        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        // Initial decode
        ctx.decode(&mut batch).context("llama_decode() failed")?;

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let t_main_start = ggml_time_us();

        while n_cur <= n_len {
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if model.is_eog_token(token) {
                break;
            }

            let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
            let output_str = String::from_utf8(output_bytes)?;
            on_token(&output_str);
            output.push_str(&output_str);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch).context("Failed to eval")?;
            n_decode += 1;
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

        println!(
            "\nDecoded {} tokens in {:.2} s, speed {:.2} t/s",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        println!("{}", ctx.timings());

        Ok(output)
    }
}

/// Commands to send to our dedicated thread
pub enum LlamaCommand {
    /// Load a new model path
    LoadModel {
        model_path: String,
        reply: oneshot::Sender<Result<()>>,
    },
    /// Generate text for a prompt (full output)
    Generate {
        prompt: String,
        reply: oneshot::Sender<Result<String>>,
    },
    /// Streaming generation: send tokens on `token_tx`, then send Ok/Err on `reply`
    GenerateStream {
        prompt: String,
        token_tx: tokio_mpsc::Sender<String>,
        reply: oneshot::Sender<Result<()>>,
    },
}

/// A handle you can use to send commands to the single-threaded actor.
/// These methods are async only because they await the `oneshot` reply.
#[derive(Clone)]
pub struct LlamaActorHandle {
    /// The standard library channel to our dedicated thread
    tx: Sender<LlamaCommand>,
}

impl LlamaActorHandle {
    pub fn new(tx: Sender<LlamaCommand>) -> Self {
        Self { tx }
    }

    /// Tell the actor to load a model. Await the Result.
    pub async fn load_model(&self, model_path: String) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = LlamaCommand::LoadModel {
            model_path,
            reply: reply_tx,
        };
        // Send the command (blocking send)
        self.tx.send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread has closed"))?;
        // Await the reply
        reply_rx.await?
    }

    /// Request a blocking generation, returning the final text
    pub async fn generate(&self, prompt: String) -> Result<String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = LlamaCommand::Generate {
            prompt,
            reply: reply_tx,
        };
        self.tx.send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread has closed"))?;
        reply_rx.await?
    }

    /// Request streaming generation. Return a receiver that yields tokens as they appear.
    pub async fn generate_stream(&self, prompt: String) -> Result<tokio_mpsc::Receiver<String>> {
        // Where we’ll push tokens
        let (token_tx, token_rx) = tokio_mpsc::channel(32);

        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = LlamaCommand::GenerateStream {
            prompt,
            token_tx,
            reply: reply_tx,
        };
        self.tx.send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread has closed"))?;
        // Return the token receiver immediately. 
        // The final success/failure is in the `reply_rx` if you want to await it.
        Ok(token_rx)
    }
}

/// Start the Llama “actor” in a dedicated std thread. 
/// The returned handle can be cloned and used from any async context.
pub fn start_llama_thread(
    initial_config: LlamaConfig,
) -> (LlamaActorHandle, thread::JoinHandle<()>) {
    // std::sync::mpsc channel for commands
    let (cmd_tx, cmd_rx) = mpsc::channel::<LlamaCommand>();

    // Spawn the single dedicated thread
    let handle = std::thread::spawn(move || {
        // Build LlamaRunner *inside* the thread
        let mut runner = LlamaRunner::new(initial_config);

        // Actor loop: read commands, handle them
        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                LlamaCommand::LoadModel { model_path, reply } => {
                    runner.config.model_path = model_path;
                    let res = runner.load_model();
                    let _ = reply.send(res);
                }
                LlamaCommand::Generate { prompt, reply } => {
                    let res = runner.generate_blocking(&prompt, |_| {});
                    let _ = reply.send(res);
                }
                LlamaCommand::GenerateStream { prompt, token_tx, reply } => {
                    let res = runner.generate_blocking(&prompt, |token| {
                        // For each token, push it into the async channel
                        // If the receiver is closed, ignore the error
                        let _ = token_tx.blocking_send(token.to_string());
                    });
                    // Map Ok(...) => Ok(()) so the caller sees success or failure
                    let _ = reply.send(res.map(|_| ()));
                }
            }
        }

        eprintln!("Llama actor thread: command channel closed, exiting.");
    });

    // Return a handle so user can send commands
    (LlamaActorHandle::new(cmd_tx), handle)
}
