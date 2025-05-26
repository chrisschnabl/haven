use std::sync::mpsc::{self, Sender, Receiver};
use std::thread;
use tokio::sync::oneshot;
use tracing::{instrument, info, error};
#[cfg(feature = "use_rust_bert")]
use crate::runner::BertRunner;
#[cfg(not(feature = "use_rust_bert"))]
use crate::mock_runner::BertRunner;
use crate::label::Label;
use crate::BertRunnerTrait;
use std::path::PathBuf;

pub enum BertCommand {
    LoadModel {
        reply: oneshot::Sender<anyhow::Result<()>>,
    },
    Predict {
        input: Vec<String>,
        reply: oneshot::Sender<anyhow::Result<Vec<Label>>>,
    },
}

#[derive(Clone)]
pub struct BertActorHandle {
    cmd_tx: Sender<BertCommand>,
}

impl BertActorHandle {
    pub fn new(cmd_tx: Sender<BertCommand>) -> Self {
        Self { cmd_tx }
    }

    #[instrument(skip(self))]
    pub async fn load_model(&self) -> anyhow::Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = BertCommand::LoadModel { reply: reply_tx };
        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread closed"))?;
        reply_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped reply"))?
    }

    #[instrument(skip(self), fields(input_len = input.len()))]
    pub async fn predict(&self, input: Vec<String>) -> anyhow::Result<Vec<Label>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = BertCommand::Predict { input, reply: reply_tx };
        self.cmd_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("Actor thread closed"))?;
        reply_rx.await.map_err(|_| anyhow::anyhow!("Actor dropped reply"))?
    }
}

#[instrument]
pub fn start_bert_actor() -> (BertActorHandle, thread::JoinHandle<()>) {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let handle = BertActorHandle::new(cmd_tx);

    let join_handle = thread::spawn(move || {
        let mut runner = BertRunner::new(PathBuf::from("rust_model.ot"), PathBuf::from("config.json"), PathBuf::from("vocab.txt"));
        actor_loop(&mut runner, cmd_rx);
        error!("BERT actor thread: command channel closed, exiting.");
    });

    (handle, join_handle)
}

#[instrument(skip(runner, cmd_rx))]
fn actor_loop(runner: &mut BertRunner, cmd_rx: Receiver<BertCommand>) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            BertCommand::LoadModel { reply } => {
                let res = runner.load_model();
                let _ = reply.send(res);
            }
            BertCommand::Predict { input, reply } => {
                info!("Processing prediction request for {} inputs", input.len());
                let res = runner.predict(input);
                let _ = reply.send(res);
            }
        }
    }
}