// src/client.rs

use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockStream};
use tracing::{info, error, instrument};
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use std::path::Path;
use std::marker::PhantomData;
use indicatif::{ProgressBar, ProgressStyle};

use crate::vsock::{write_message, read_message, Message, Operation};

// Client state traits
pub trait ClientState {}

// Client concrete states
pub struct Disconnected;
pub struct Connected;
pub struct LlamaSent;
pub struct BertSent;
pub struct DatasetSent;
pub struct EvaluationComplete;
pub struct AttestationReceived;

impl ClientState for Disconnected {}
impl ClientState for Connected {}
impl ClientState for LlamaSent {}
impl ClientState for BertSent {}
impl ClientState for DatasetSent {}
impl ClientState for EvaluationComplete {}
impl ClientState for AttestationReceived {}

// Shared state that persists across all client state transitions
pub struct ClientSharedState {
    stream: Option<VsockStream>,
    llama_path: Option<String>,
    bert_path: Option<String>,
    dataset_path: Option<String>,
    attestation: Option<Vec<u8>>,
}

// Main Client struct with generic state parameter
pub struct ModelClient<S: ClientState> {
    shared: ClientSharedState,
    state: PhantomData<S>,
}

// Implementation for the disconnected state
impl ModelClient<Disconnected> {
    /// Creates a new ModelClient in the disconnected state
    pub fn new() -> Self {
        ModelClient {
            shared: ClientSharedState {
                stream: None,
                llama_path: None,
                bert_path: None,
                dataset_path: None,
                attestation: None,
            },
            state: PhantomData,
        }
    }

    /// Connect to the server, transitioning to Connected state
    #[instrument(skip(self))]
    pub async fn connect(mut self, cid: u32, port: u32) -> Result<ModelClient<Connected>> {
        info!("Connecting to server at CID {}, port {}", cid, port);
        
        let addr = VsockAddr::new(cid, port);
        let stream = VsockStream::connect(addr)
            .await
            .context("Failed to connect to server")?;
        
        info!("Successfully connected to server");
        
        // Store the stream and transition to Connected state
        self.shared.stream = Some(stream);
        
        Ok(ModelClient {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Helper function to send a file through the vsock stream
async fn send_file(stream: &mut VsockStream, file_path: &str) -> Result<()> {
    let path = Path::new(file_path);
    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path);
    
    let mut file = File::open(path)
        .await
        .context(format!("Failed to open file: {}", file_path))?;
    
    let file_metadata = tokio::fs::metadata(path).await?;
    let total_size = file_metadata.len();
    let mut total_sent = 0;
    
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .expect("Failed to set template for progress bar")
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Transferring '{}'...", file_name));   

    let mut buf = vec![0u8; 4096]; // Reasonable buffer size for chunk transfer

    loop {
        let len = file.read(&mut buf).await?;
        if len == 0 {
            // EOF => send EofFile
            let msg = Message {
                op: Operation::EofFile,
                file_path: Some(file_path.to_string()),
                data: vec![],
            };
            write_message(stream, &msg).await?;
            pb.finish_with_message("File transfer complete. Sent EOF marker.");
            break;
        }

        let msg = Message {
            op: Operation::SendFile,
            file_path: Some(file_path.to_string()),
            data: buf[..len].to_vec(),
        };
        write_message(stream, &msg).await?;

        total_sent += len as u64;
        pb.set_position(total_sent);
    }
    
    Ok(())
}

// Implementation for the connected state
impl ModelClient<Connected> {
    /// Send the LLaMA model file to the server
    #[instrument(skip(self))]
    pub async fn send_llama_model(mut self, file_path: &str) -> Result<ModelClient<LlamaSent>> {
        info!("Sending LLaMA model file: {}", file_path);
        
        let stream = self.shared.stream.as_mut()
            .context("Stream not connected")?;
        
        send_file(stream, file_path).await?;
        
        // Store the file path and transition to LlamaSent state
        self.shared.llama_path = Some(file_path.to_string());
        
        Ok(ModelClient {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when LLaMA model is sent
impl ModelClient<LlamaSent> {
    /// Send the BERT model file to the server
    #[instrument(skip(self))]
    pub async fn send_bert_model(mut self, file_path: &str) -> Result<ModelClient<BertSent>> {
        info!("Sending BERT model file: {}", file_path);
        
        let stream = self.shared.stream.as_mut()
            .context("Stream not connected")?;
        
        send_file(stream, file_path).await?;
        
        // Store the file path and transition to BertSent state
        self.shared.bert_path = Some(file_path.to_string());
        
        Ok(ModelClient {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when BERT model is sent
impl ModelClient<BertSent> {
    /// Send the evaluation dataset to the server
    #[instrument(skip(self))]
    pub async fn send_dataset(mut self, file_path: &str) -> Result<ModelClient<DatasetSent>> {
        info!("Sending evaluation dataset: {}", file_path);
        
        let stream = self.shared.stream.as_mut()
            .context("Stream not connected")?;
        
        send_file(stream, file_path).await?;
        
        // Store the file path and transition to DatasetSent state
        self.shared.dataset_path = Some(file_path.to_string());
        
        Ok(ModelClient {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when dataset is sent
impl ModelClient<DatasetSent> {
    /// Wait for evaluation to complete
    #[instrument(skip(self))]
    pub async fn wait_for_evaluation(mut self) -> Result<ModelClient<EvaluationComplete>> {
        info!("Waiting for server to complete evaluation...");
        
        let stream = self.shared.stream.as_mut()
            .context("Stream not connected")?;
        
        // Wait for progress messages until evaluation is complete
        loop {
            let msg = read_message(stream).await?;
            
            match msg.op {
                Operation::Progress => {
                    let progress = String::from_utf8_lossy(&msg.data);
                    info!("Server progress: {}", progress);
                }
                
                Operation::Attestation => {
                    // When we receive an attestation, evaluation is complete
                    info!("Received attestation data from server");
                    self.shared.attestation = Some(msg.data);
                    break;
                }
                
                _ => {
                    error!("Unexpected message type: {:?}", msg.op);
                }
            }
        }
        
        // Transition to EvaluationComplete state
        Ok(ModelClient {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when evaluation is complete
impl ModelClient<EvaluationComplete> {
    /// Verify the attestation received from the server
    #[instrument(skip(self))]
    pub async fn verify_attestation(mut self) -> Result<ModelClient<AttestationReceived>> {
        info!("Verifying attestation from server...");
        
        let attestation = self.shared.attestation.as_ref()
            .context("No attestation data received")?;
        
        // In a real application, we would verify the attestation here
        // For now, just print it
        let attestation_str = String::from_utf8_lossy(attestation);
        info!("Attestation data: {}", attestation_str);
        
        // Simulate verification
        if attestation_str.starts_with("ATTESTATION-DATA") {
            info!("Attestation verification successful");
        } else {
            return Err(anyhow::anyhow!("Invalid attestation format"));
        }
        
        // Transition to AttestationReceived state
        Ok(ModelClient {
            shared: self.shared,
            state: PhantomData,
        })
    }
}

// Implementation for when attestation is verified
impl ModelClient<AttestationReceived> {
    /// Complete the client session
    #[instrument(skip(self))]
    pub async fn complete_session(mut self) -> Result<()> {
        info!("Waiting for final completion message from server...");
        
        let stream = self.shared.stream.as_mut()
            .context("Stream not connected")?;
        
        // Wait for completion message
        let msg = read_message(stream).await?;
        
        match msg.op {
            Operation::Complete => {
                let completion = String::from_utf8_lossy(&msg.data);
                info!("Server session complete: {}", completion);
            }
            
            _ => {
                error!("Unexpected final message: {:?}", msg.op);
            }
        }
        
        info!("Client session completed successfully");
        Ok(())
    }
}

// Main client entry point
#[instrument]
pub async fn run_client(
    cid: u32, 
    port: u32, 
    llama_path: &str, 
    bert_path: &str, 
    dataset_path: &str
) -> Result<()> {
    info!("Starting client for model evaluation workflow");
    
    let client = ModelClient::new();
    
    // Execute the full workflow using typestate transitions
    let client = client.connect(cid, port).await?;
    info!("Connected to server");
    
    let client = client.send_llama_model(llama_path).await?;
    info!("LLaMA model sent");
    
    let client = client.send_bert_model(bert_path).await?;
    info!("BERT model sent");
    
    let client = client.send_dataset(dataset_path).await?;
    info!("Evaluation dataset sent");
    
    let client = client.wait_for_evaluation().await?;
    info!("Evaluation completed");
    
    let client = client.verify_attestation().await?;
    info!("Attestation verified");
    
    client.complete_session().await?;
    info!("Client session completed");
    
    Ok(())
}