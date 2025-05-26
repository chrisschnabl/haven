use anyhow::Result;
use std::thread;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use std::sync::mpsc::{self, Sender, Receiver};
use tracing::{error, info, debug, instrument};

use crate::runner::LlamaRunner;
use crate::config::LlamaConfig;

/// The commands our dedicated thread can handle
pub enum LlamaCommand {
    LoadModel {
        model_path: String,
        reply: oneshot::Sender<Result<()>>,
    },
    Generate {
        prompt: String,
        token_tx: tokio_mpsc::Sender<String>,
        reply: oneshot::Sender<Result<()>>,
    },
}

/// An actor handle you can clone to send commands from async code.
#[derive(Clone)]
pub struct LlamaActorHandle {
    pub(crate) cmd_tx: Sender<LlamaCommand>,
}

impl LlamaActorHandle {
    pub fn new(cmd_tx: Sender<LlamaCommand>) -> Self {
        Self { cmd_tx }
    }

    #[instrument(skip(self), fields(model_path = model_path))]
    pub async fn load_model(&self, model_path: String) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = LlamaCommand::LoadModel {
            model_path,
            reply: reply_tx,
        };
        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread closed"))?;

        reply_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped reply"))?
    }

    #[instrument(skip(self), fields(prompt_len = prompt.len()))]
    pub async fn generate(&self, prompt: String) -> Result<String> {
        let (token_tx, mut token_rx) = tokio_mpsc::channel(64);
        let (reply_tx, reply_rx) = oneshot::channel();

        let cmd = LlamaCommand::Generate {
            prompt,
            token_tx,
            reply: reply_tx,
        };
        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread closed"))?;

        // Collect tokens
        let mut output = String::new();
        while let Some(token) = token_rx.recv().await {
            output.push_str(&token);
        }

        reply_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped reply"))??;

        Ok(output)
    }

    #[instrument(skip(self), fields(prompt_len = prompt.len()))]
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
            .map_err(|_| anyhow::anyhow!("Actor thread closed"))?;

        // Return the token receiver + final result
        Ok((token_rx, reply_rx))
    }
}

#[instrument(skip(config), fields(threads = config.threads, context_size = config.context_size.get()))]
pub fn start_llama_thread(config: LlamaConfig) -> (LlamaActorHandle, thread::JoinHandle<()>) {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let handle = LlamaActorHandle::new(cmd_tx);

    // Start a dedicated thread here that owns the runner
    let join_handle = thread::spawn(move || {
        let mut runner = LlamaRunner::new(config);
        actor_loop(&mut runner, cmd_rx);
        info!("Actor thread: command channel closed, exiting.");
    });

    (handle, join_handle)
}

#[instrument(skip(runner, cmd_rx))]
fn actor_loop(runner: &mut LlamaRunner, cmd_rx: Receiver<LlamaCommand>) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            LlamaCommand::LoadModel { model_path, reply } => {
                runner.config.model_path = Some(model_path.clone());
                if runner.is_model_loaded() {
                    info!("Model already loaded; skipping load.");
                    let _ = reply.send(Ok(()));
                } else {
                    info!("Loading model from path: {}", model_path);
                    let res = runner.load_model();
                    if let Err(e) = &res {
                        error!("Load model error: {:?}", e);
                    }
                    let _ = reply.send(res);
                }
            }
            LlamaCommand::Generate {
                prompt,
                token_tx,
                reply,
            } => {
                info!("Generating response for prompt of length {}", prompt.len());
                let result = runner.generate_blocking(&prompt, |token_str| {
                    // This is blocking send because we're on a dedicated thread
                    let _ = token_tx.blocking_send(token_str.to_string());
                });
                drop(token_tx); // close channel

                match result {
                    Ok((n_decode, duration, tokenize_duration, prompt_duration, prompt_tokens)) => {
                        debug!("Generated {} tokens in {:?} at {:.2} tokens/sec, tokenize: {:?}, prompt: {:?}, prompt_tokens: {}", 
                              n_decode, duration, n_decode as f64 / duration.as_secs_f64(), tokenize_duration, prompt_duration, prompt_tokens);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        error!("Generation error: {:?}", e);
                        let _ = reply.send(Err(e));
                    }
                }
            }
        }
    }
}