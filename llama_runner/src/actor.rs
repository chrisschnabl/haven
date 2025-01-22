use anyhow::Result;
use std::thread;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use std::sync::mpsc::{self, Sender, Receiver};
use tracing::{error, info};

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

    /// Load a model, awaiting the final result
    pub async fn load_model(&self, model_path: String) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = LlamaCommand::LoadModel {
            model_path,
            reply: reply_tx,
        };
        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread closed"))?;

        // Wait for the final result
        reply_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped reply"))?
    }

    /// Generate the entire output. Collect tokens into a single String.
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

        // Check final success/failure
        reply_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped reply"))??;

        Ok(output)
    }

    /// Generate text as a stream: (Receiver, final oneshot)
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

/// Start a dedicated thread that owns `LlamaRunner`.
/// We do not require `Send` for the runner because it never leaves that thread.
pub fn start_llama_thread(config: LlamaConfig) -> (LlamaActorHandle, thread::JoinHandle<()>) {
    // We use std::sync::mpsc for the commands
    let (cmd_tx, cmd_rx) = mpsc::channel();

    // The user handle
    let handle = LlamaActorHandle::new(cmd_tx);

    // The dedicated thread
    let join_handle = thread::spawn(move || {
        let mut runner = LlamaRunner::new(config);
        actor_loop(&mut runner, cmd_rx);
        info!("Actor thread: command channel closed, exiting.");
    });

    (handle, join_handle)
}

fn actor_loop(runner: &mut LlamaRunner, cmd_rx: Receiver<LlamaCommand>) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            LlamaCommand::LoadModel { model_path, reply } => {
                runner.config.model_path = Some(model_path);
                let res = runner.load_model();
                if let Err(e) = &res {
                    error!("Load model error: {:?}", e);
                }
                let _ = reply.send(res);
            }
            LlamaCommand::Generate {
                prompt,
                token_tx,
                reply,
            } => {
                let res = runner.generate_blocking(&prompt, |token_str| {
                    // This is blocking send because we're on a dedicated thread
                    let _ = token_tx.blocking_send(token_str.to_string());
                });
                drop(token_tx); // close channel

                if let Err(e) = &res {
                    error!("Generate error: {:?}", e);
                }
                let _ = reply.send(res);
            }
        }
    }
}