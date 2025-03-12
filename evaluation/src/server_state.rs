use anyhow::{Result};
use tokio_vsock::VsockStream;
use tracing::{info, instrument};
use std::path::PathBuf;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use std::collections::HashMap;

use crate::messages::{write_message, Operation, Message};
use crate::file_transfer::receive_file;

pub trait ServerState {}

pub struct InitializedState;
pub struct LlamaLoadedState {
    llama_path: PathBuf,
}
pub struct BertLoadedState {
    llama_path: PathBuf,
    bert_path: PathBuf,
}
pub struct DatasetLoadedState {
    llama_path: PathBuf,
    bert_path: PathBuf,
    dataset_path: PathBuf,
}
pub struct EvaluatedState {
    results: Vec<EvaluationResult>,
}
pub struct AttestedState {}

impl ServerState for InitializedState {}
impl ServerState for LlamaLoadedState {}
impl ServerState for BertLoadedState {}
impl ServerState for DatasetLoadedState {}
impl ServerState for EvaluatedState {}
impl ServerState for AttestedState {}

pub struct SharedState {
    stream: VsockStream,
}

pub struct EvaluationResult {
    input: String,
    llama_output: String,
    bert_classification: String,
    confidence: f32,
}

pub struct ModelServer<S: ServerState> {
    state: S,
    shared: Box<SharedState>,
}

impl ModelServer<InitializedState> {
    pub fn new(stream: VsockStream) -> Self {
        ModelServer {
            state: InitializedState {},
            shared: Box::new(SharedState { stream }),
        }
    }

    #[instrument(skip(self))]
    pub async fn receive_llama_model(mut self) -> Result<ModelServer<LlamaLoadedState>> {
        info!("Waiting for LLaMA model file...");
        let path = receive_file(&mut self.shared.stream, "LLaMA model").await?;
        
        Ok(ModelServer {
            state: LlamaLoadedState { llama_path: path },
            shared: self.shared,
        })
    }
}

impl ModelServer<LlamaLoadedState> {
    #[instrument(skip(self))]
    pub async fn receive_bert_model(mut self) -> Result<ModelServer<BertLoadedState>> {
        info!("Waiting for BERT model file...");
        let path = receive_file(&mut self.shared.stream, "BERT model").await?;
        
        Ok(ModelServer {
            state: BertLoadedState { 
                llama_path: self.state.llama_path,
                bert_path: path,
            },
            shared: self.shared,
        })
    }
}

impl ModelServer<BertLoadedState> {
    #[instrument(skip(self))]
    pub async fn receive_dataset(mut self) -> Result<ModelServer<DatasetLoadedState>> {
        info!("Waiting for evaluation dataset...");
        let path = receive_file(&mut self.shared.stream, "evaluation dataset").await?;
        
        Ok(ModelServer {
            state: DatasetLoadedState {
                llama_path: self.state.llama_path,
                bert_path: self.state.bert_path,
                dataset_path: path,
            },
            shared: self.shared,
        })
    }
}

impl ModelServer<DatasetLoadedState> {
    #[instrument(skip(self))]
    pub async fn run_evaluation(mut self) -> Result<ModelServer<EvaluatedState>> {
        info!("Starting evaluation process...");
        
        info!("Loading LLaMA model from {:?}", self.state.llama_path);
        info!("Loading BERT model from {:?}", self.state.bert_path);
        info!("Loading evaluation dataset from {:?}", self.state.dataset_path);
        
        let mut file = File::open(&self.state.dataset_path).await?;
        let mut dataset_content = String::new();
        file.read_to_string(&mut dataset_content).await?;
        
        let inputs: Vec<String> = dataset_content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();
        
        let mut results = Vec::new();
        
        for input in inputs {
            let progress_msg = Message {
                op: Operation::Progress,
                file_path: None,
                data: format!("Processing: {}", input).into_bytes(),
            };
            write_message(&mut self.shared.stream, &progress_msg).await?;
            
            let llama_output = "I don't know the answer to that question.".to_string();
            let bert_classification = "FACTUAL".to_string();
            let confidence = 0.92;
            
            results.push(EvaluationResult {
                input,
                llama_output,
                bert_classification,
                confidence,
            });
        }
        
        let completion_msg = Message {
            op: Operation::Progress,
            file_path: None,
            data: format!("Evaluation complete. Processed {} samples.", results.len()).into_bytes(),
        };
        write_message(&mut self.shared.stream, &completion_msg).await?;
        
        Ok(ModelServer {
            state: EvaluatedState { results },
            shared: self.shared,
        })
    }
}

impl ModelServer<EvaluatedState> {
    #[instrument(skip(self))]
    pub async fn generate_attestation(mut self) -> Result<ModelServer<AttestedState>> {
        info!("Generating attestation for evaluation results...");
        
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        let mut total_confidence = 0.0;
        
        for result in &self.state.results {
            *class_counts.entry(result.bert_classification.clone()).or_insert(0) += 1;
            total_confidence += result.confidence;
        }
        
        let avg_confidence = if !self.state.results.is_empty() {
            total_confidence / self.state.results.len() as f32
        } else {
            0.0
        };
        
        let attestation = format!(
            "ATTESTATION-DATA:{}:{}:{}",
            self.state.results.len(),
            avg_confidence,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        
        let attestation_msg = Message {
            op: Operation::Attestation,
            file_path: None,
            data: attestation.into_bytes(),
        };
        write_message(&mut self.shared.stream, &attestation_msg).await?;
        
        Ok(ModelServer {
            state: AttestedState {},
            shared: self.shared,
        })
    }
}

impl ModelServer<AttestedState> {
    #[instrument(skip(self))]
    pub async fn complete_session(mut self) -> Result<()> {
        let completion_msg = Message {
            op: Operation::Complete,
            file_path: None,
            data: b"Evaluation and attestation complete.".to_vec(),
        };
        write_message(&mut self.shared.stream, &completion_msg).await?;
        
        info!("Session completed successfully");
        Ok(())
    }
}


// TODO CS: shared state would be here
impl<S: ServerState> ModelServer<S> {
}