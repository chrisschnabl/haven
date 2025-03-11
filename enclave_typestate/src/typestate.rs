
use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tracing::{error, info, instrument};
use std::path::PathBuf;
use std::marker::PhantomData;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::collections::HashMap;

use crate::vsock::{read_message, write_message, Operation, Message};

pub trait State {}

pub struct Initialized;
pub struct LlamaModelLoaded;
pub struct BertModelLoaded;
pub struct DatasetLoaded;
pub struct Evaluated;
pub struct Attested;

impl State for Initialized {}
impl State for LlamaModelLoaded {}
impl State for BertModelLoaded {}
impl State for DatasetLoaded {}
impl State for Evaluated {}
impl State for Attested {}

pub struct SharedState {
    stream: VsockStream,
    llama_model_path: Option<PathBuf>,
    bert_model_path: Option<PathBuf>,
    dataset_path: Option<PathBuf>,
    evaluation_results: Vec<EvaluationResult>,
    attestation_data: Option<Vec<u8>>,
}

pub struct EvaluationResult {
    input: String,
    llama_output: String,
    bert_classification: String,
    confidence: f32,
}

pub struct ModelServer<S: State> {
    shared: SharedState,
    state: PhantomData<S>,
}

impl<S: State> ModelServer<S> {
    pub fn log(&self, message: &str) {
        info!("{}", message);
    }
    
    pub fn stream(&mut self) -> &mut VsockStream {
        &mut self.shared.stream
    }
}

impl ModelServer<Initialized> {
    pub fn new(stream: VsockStream) -> Self {
        ModelServer {
            shared: SharedState {
                stream,
                llama_model_path: None,
                bert_model_path: None,
                dataset_path: None,
                evaluation_results: Vec::new(),
                attestation_data: None,
            },
            state: PhantomData,
        }
    }

    #[instrument(skip(self))]
    pub async fn receive_llama_model(mut self) -> Result<ModelServer<LlamaModelLoaded>> {
        info!("Waiting for LLaMA model file...");
        
        let mut file_path = None;
        let mut file: Option<File> = None;
        let mut total_received = 0u64;
        
        loop {
            let msg = read_message(&mut self.shared.stream).await
                .context("Failed to read message while receiving LLaMA model")?;
            
            match msg.op {
                Operation::SendFile => {
                    if file_path.is_none() && msg.file_path.is_some() {
                        file_path = msg.file_path.clone();
                        info!("Receiving LLaMA model: {:?}", file_path);
                        
                        // Create the file for writing
                        if let Some(path) = &file_path {
                            file = Some(File::create(path).await?);
                        }
                    }
                    
                    // Write data to file if we have it
                    if let Some(ref mut f) = file {
                        f.write_all(&msg.data).await?;
                        total_received += msg.data.len() as u64;
                        info!("Received {} bytes of LLaMA model data", total_received);
                    }
                }
                
                Operation::EofFile => {
                    info!("LLaMA model file transfer complete. Received {} bytes", total_received);
                    break;
                }
                
                _ => {
                    return Err(anyhow::anyhow!("Unexpected operation during LLaMA model transfer: {:?}", msg.op));
                }
            }
        }
        
        // Save the file path for later use
        if let Some(path) = file_path {
            self.shared.llama_model_path = Some(PathBuf::from(path));
        } else {
            return Err(anyhow::anyhow!("No file path received for LLaMA model"));
        }
        
        // Transition to LlamaModelLoaded state
        Ok(ModelServer {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when the LLaMA model is loaded
impl ModelServer<LlamaModelLoaded> {
    /// Receives and saves the BERT model, transitioning to BertModelLoaded state
    #[instrument(skip(self))]
    pub async fn receive_bert_model(mut self) -> Result<ModelServer<BertModelLoaded>> {
        info!("Waiting for BERT model file...");
        
        let mut file_path = None;
        let mut file: Option<File> = None;
        let mut total_received = 0u64;
        
        loop {
            let msg = read_message(&mut self.shared.stream).await
                .context("Failed to read message while receiving BERT model")?;
            
            match msg.op {
                Operation::SendFile => {
                    if file_path.is_none() && msg.file_path.is_some() {
                        file_path = msg.file_path.clone();
                        info!("Receiving BERT model: {:?}", file_path);
                        
                        // Create the file for writing
                        if let Some(path) = &file_path {
                            file = Some(File::create(path).await?);
                        }
                    }
                    
                    // Write data to file if we have it
                    if let Some(ref mut f) = file {
                        f.write_all(&msg.data).await?;
                        total_received += msg.data.len() as u64;
                        info!("Received {} bytes of BERT model data", total_received);
                    }
                }
                
                Operation::EofFile => {
                    info!("BERT model file transfer complete. Received {} bytes", total_received);
                    break;
                }
                
                _ => {
                    return Err(anyhow::anyhow!("Unexpected operation during BERT model transfer: {:?}", msg.op));
                }
            }
        }
        
        // Save the file path
        if let Some(path) = file_path {
            self.shared.bert_model_path = Some(PathBuf::from(path));
        } else {
            return Err(anyhow::anyhow!("No file path received for BERT model"));
        }
        
        // Transition to BertModelLoaded state
        Ok(ModelServer {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when both models are loaded
impl ModelServer<BertModelLoaded> {
    /// Receives the evaluation dataset, transitioning to DatasetLoaded state
    #[instrument(skip(self))]
    pub async fn receive_evaluation_dataset(mut self) -> Result<ModelServer<DatasetLoaded>> {
        info!("Waiting for evaluation dataset...");
        
        let mut file_path = None;
        let mut file: Option<File> = None;
        let mut total_received = 0u64;
        
        loop {
            let msg = read_message(&mut self.shared.stream).await
                .context("Failed to read message while receiving evaluation dataset")?;
            
            match msg.op {
                Operation::SendFile => {
                    if file_path.is_none() && msg.file_path.is_some() {
                        file_path = msg.file_path.clone();
                        info!("Receiving evaluation dataset: {:?}", file_path);
                        
                        // Create the file for writing
                        if let Some(path) = &file_path {
                            file = Some(File::create(path).await?);
                        }
                    }
                    
                    // Write data to file if we have it
                    if let Some(ref mut f) = file {
                        f.write_all(&msg.data).await?;
                        total_received += msg.data.len() as u64;
                        info!("Received {} bytes of evaluation dataset", total_received);
                    }
                }
                
                Operation::EofFile => {
                    info!("Evaluation dataset transfer complete. Received {} bytes", total_received);
                    break;
                }
                
                _ => {
                    return Err(anyhow::anyhow!("Unexpected operation during evaluation dataset transfer: {:?}", msg.op));
                }
            }
        }
        
        // Save the dataset path
        if let Some(path) = file_path {
            self.shared.dataset_path = Some(PathBuf::from(path));
        } else {
            return Err(anyhow::anyhow!("No file path received for evaluation dataset"));
        }
        
        // Transition to DatasetLoaded state
        Ok(ModelServer {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when dataset is loaded and ready for evaluation
impl ModelServer<DatasetLoaded> {
    /// Run inference on the dataset using both models
    #[instrument(skip(self))]
    pub async fn run_evaluation(mut self) -> Result<ModelServer<Evaluated>> {
        info!("Starting evaluation process...");
        
        let llama_path = self.shared.llama_model_path.as_ref()
            .context("LLaMA model path not found")?;
        let bert_path = self.shared.bert_model_path.as_ref()
            .context("BERT model path not found")?;
        let dataset_path = self.shared.dataset_path.as_ref()
            .context("Dataset path not found")?;


        info!("Loading LLaMA model from {:?}", llama_path);
        // let llama_model = load_llama_model(llama_path).await?;
        
        // 2. Load the BERT model
        info!("Loading BERT model from {:?}", bert_path);
        // let bert_model = load_bert_model(bert_path).await?;
        
        // 3. Load the dataset
        info!("Loading evaluation dataset from {:?}", dataset_path);
        // let dataset = load_dataset(dataset_path).await?;
        
        // For demonstration, let's simulate loading a dataset by reading the file
        let mut file = File::open(dataset_path).await?;
        let mut dataset_content = String::new();
        file.read_to_string(&mut dataset_content).await?;
        
        // Parse dataset content (simplified - in real code this would be more robust)
        let inputs: Vec<String> = dataset_content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();
        
        // Process each input and collect results
        for input in inputs {
            // Send progress update to client
            let progress_msg = Message {
                op: Operation::Progress,
                file_path: None,
                data: format!("Processing: {}", input).into_bytes(),
            };
            write_message(&mut self.shared.stream, &progress_msg).await?;
            
            // Simulate LLaMA inference
            info!("Running LLaMA inference on: {}", input);
            let llama_output = match input.to_lowercase() {
                s if s.contains("capital") => "Paris is the capital of France.".to_string(),
                s if s.contains("quantum") => "Quantum computing uses quantum bits that can be both 0 and 1...".to_string(),
                s if s.contains("photo") => "Photosynthesis is a process used by plants to convert light energy into chemical energy...".to_string(),
                _ => "I don't know the answer to that question.".to_string(),
            };
            
            // Simulate BERT classification
            info!("Running BERT classification on LLaMA output");
            let bert_classification = if llama_output.contains("capital") {
                "FACTUAL".to_string()
            } else if llama_output.contains("quantum") {
                "TECHNICAL".to_string()
            } else {
                "DESCRIPTIVE".to_string()
            };
            
            let confidence = 0.92; // Simulated confidence score
            
            // Store the result
            self.shared.evaluation_results.push(EvaluationResult {
                input,
                llama_output,
                bert_classification,
                confidence,
            });
        }
        
        // Send completion message to client
        let completion_msg = Message {
            op: Operation::Progress,
            file_path: None,
            data: format!("Evaluation complete. Processed {} samples.", 
                          self.shared.evaluation_results.len()).into_bytes(),
        };
        write_message(&mut self.shared.stream, &completion_msg).await?;
        
        info!("Evaluation process complete");
        
        // Transition to Evaluated state
        Ok(ModelServer {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when evaluation is complete
impl ModelServer<Evaluated> {
    /// Generate an attestation for the evaluation results
    #[instrument(skip(self))]
    pub async fn generate_attestation(mut self) -> Result<ModelServer<Attested>> {
        info!("Generating attestation for evaluation results...");
        
        // Aggregate results
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        let mut total_confidence = 0.0;
        
        for result in &self.shared.evaluation_results {
            *class_counts.entry(result.bert_classification.clone()).or_insert(0) += 1;
            total_confidence += result.confidence;
        }
        
        let avg_confidence = if !self.shared.evaluation_results.is_empty() {
            total_confidence / self.shared.evaluation_results.len() as f32
        } else {
            0.0
        };
        
        // Create summary string
        let summary = format!(
            "Evaluation Summary:\n\
             Total samples: {}\n\
             Average confidence: {:.2}%\n\
             Class distribution: {:?}\n\
             LLaMA model: {:?}\n\
             BERT model: {:?}",
            self.shared.evaluation_results.len(),
            avg_confidence * 100.0,
            class_counts,
            self.shared.llama_model_path,
            self.shared.bert_model_path
        );
        
        info!("{}", summary);
        
        // Simulate generating an attestation
        // In a real implementation, this would call into a TEE attestation system
        let attestation = format!(
            "ATTESTATION-DATA:{}:{}:{}",
            self.shared.evaluation_results.len(),
            avg_confidence,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        
        // Store the attestation
        self.shared.attestation_data = Some(attestation.into_bytes());
        
        // Send attestation to client
        let attestation_msg = Message {
            op: Operation::Attestation,
            file_path: None,
            data: self.shared.attestation_data.clone().unwrap(),
        };
        write_message(&mut self.shared.stream, &attestation_msg).await?;
        
        info!("Attestation generated and sent to client");
        
        // Transition to Attested state
        Ok(ModelServer {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

impl ModelServer<Attested> {
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