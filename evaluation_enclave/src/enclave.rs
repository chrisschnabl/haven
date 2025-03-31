use anyhow::{Result};
use tokio_vsock::VsockStream;
use tracing::{info, instrument};
use std::path::PathBuf;

use evaluation::{
    receive_file, write_message, send_file, Operation, Message,
    run_toxicity, analyze_toxicity,
    TaskConfig, ModelConfig, DataConfig, OutputConfig
};

pub trait ServerState {}

pub struct InitializedState;
pub struct LlamaLoadedState {
    llama_path: PathBuf,
}
pub struct BertLoadedState {
    llama_path: PathBuf,
    bert_path: PathBuf,
    config_path: PathBuf,
    vocab_path: PathBuf,
}

pub struct DatasetLoadedState {
    llama_path: PathBuf,
    bert_path: PathBuf,
    config_path: PathBuf,
    vocab_path: PathBuf,
    dataset_path: PathBuf,
}
pub struct EvaluatedState {
    results: Vec<EvaluationResult>,
    output_file: PathBuf,
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
        let config_path = receive_file(&mut self.shared.stream, "BERT model config").await?;
        let vocab_path = receive_file(&mut self.shared.stream, "BERT model vocab").await?;
        
        Ok(ModelServer {
            state: BertLoadedState { 
                llama_path: self.state.llama_path,
                bert_path: path,
                config_path,
                vocab_path,
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
                config_path: self.state.config_path,
                vocab_path: self.state.vocab_path,
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
        
        let config = TaskConfig {
            model: ModelConfig {
                model_path: self.state.llama_path,
                context_size: 8 * 1024,
                threads: 1,
                n_len: 256,
                seed: 1337,
                temp: 0.25,
                top_p: 0.7,
                skip_non_utf8: true,
                truncate_if_context_full: true,
            },
            data: DataConfig {
                dataset_path: self.state.dataset_path.to_string_lossy().to_string(),
                dataset_url: "none".to_string(), 
                limit: Some(1),
                start_from: 0,
                skip_if_longer_than: Some(1750),
            },
            output: OutputConfig {
                output_dir: PathBuf::from("."),
                file_prefix: "llama_toxic".to_string(),
            },
        };

        let output_dir = config.output.output_dir.clone();
        let file_prefix = config.output.file_prefix.clone();
        
        run_toxicity(None, None, Some(config))?;
        let responses_file = output_dir.join(&file_prefix); 
        let output_file = analyze_toxicity(&responses_file, &self.state.bert_path, &self.state.config_path, &self.state.vocab_path)?;

        // Parse results from output file
        let results = vec![EvaluationResult {
            input: "".to_string(),
            llama_output: "".to_string(),
            bert_classification: "".to_string(),
            confidence: 0.0,
        }]; // TODO: Actually parse results from output_file
        
        let completion_msg = Message {
            op: Operation::Progress,
            file_path: None,
            data: format!("Evaluation complete. Processed {} samples.", results.len()).into_bytes(),
        };
        write_message(&mut self.shared.stream, &completion_msg).await?;        
        Ok(ModelServer {
            state: EvaluatedState { 
                results,
                output_file,
            },
            shared: self.shared,
        })
    }
}

impl ModelServer<EvaluatedState> {
    #[instrument(skip(self))]
    pub async fn generate_attestation(mut self) -> Result<ModelServer<AttestedState>> {
        info!("Generating attestation for evaluation results...");
                
        let attestation = format!(
            "ATTESTATION-DATA:{}:{}",
            self.state.results.len(),
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

        // Send the results file
        send_file(&mut self.shared.stream, self.state.output_file.to_str().unwrap()).await?;
        
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