use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockStream};
use tracing::{info, error, instrument};
use std::marker::PhantomData;

use crate::messages::{read_message, Operation};
use crate::file_transfer::send_file;

pub trait ClientState {}

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

pub struct ModelClient<S: ClientState> {
    stream: Option<VsockStream>,
    attestation: Option<Vec<u8>>,
    state: PhantomData<S>,
}

impl ModelClient<Disconnected> {
    pub fn new() -> Self {
        ModelClient {
            stream: None,
            attestation: None,
            state: PhantomData,
        }
    }

    #[instrument(skip(self))]
    pub async fn connect(mut self, cid: u32, port: u32) -> Result<ModelClient<Connected>> {
        info!("Connecting to server at CID {}, port {}", cid, port);
        
        let addr = VsockAddr::new(cid, port);
        let stream = VsockStream::connect(addr)
            .await
            .context("Failed to connect to server")?;
        
        info!("Successfully connected to server");
        
        self.stream = Some(stream);
        
        Ok(ModelClient {
            stream: self.stream,
            attestation: self.attestation,
            state: PhantomData,
        })
    }
}


impl ModelClient<Connected> {
    #[instrument(skip(self))]
    pub async fn send_llama_model(mut self, file_path: &str) -> Result<ModelClient<LlamaSent>> {
        info!("Sending LLaMA model file: {}", file_path);
        
        let stream = self.stream.as_mut()
            .context("Stream not connected")?;
        
        send_file(stream, file_path).await?;
                
        Ok(ModelClient {
            stream: self.stream,
            attestation: self.attestation,
            state: PhantomData,
        })
    }
}

impl ModelClient<LlamaSent> {
    #[instrument(skip(self))]
    pub async fn send_bert_model(mut self, file_path: &str) -> Result<ModelClient<BertSent>> {
        info!("Sending BERT model file: {}", file_path);
        let stream = self.stream.as_mut()
            .context("Stream not connected")?;
        
        send_file(stream, file_path).await?;
        
        Ok(ModelClient {
            stream: self.stream,
            attestation: self.attestation,
            state: PhantomData,
        })
    }
}

impl ModelClient<BertSent> {
    #[instrument(skip(self))]
    pub async fn send_dataset(mut self, file_path: &str) -> Result<ModelClient<DatasetSent>> {
        info!("Sending dataset: {}", file_path);
        
        let stream = self.stream.as_mut()
            .context("Stream not connected")?;
        
        send_file(stream, file_path).await?;
        
        Ok(ModelClient {
            stream: self.stream,
            attestation: self.attestation,
            state: PhantomData,
        })
    }
}

impl ModelClient<DatasetSent> {
    #[instrument(skip(self))]
    pub async fn wait_for_evaluation(mut self) -> Result<ModelClient<EvaluationComplete>> {
        info!("Waiting for server to complete...");
        
        let stream = self.stream.as_mut()
            .context("Stream not connected")?;
        
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
                    self.attestation = Some(msg.data);
                    break;
                }
                
                _ => {
                    error!("Unexpected message type: {:?}", msg.op);
                }
            }
        }
        
        Ok(ModelClient {
            stream: self.stream,
            attestation: self.attestation,
            state: PhantomData,
        })
    }
}

impl ModelClient<EvaluationComplete> {
    #[instrument(skip(self))]
    pub async fn verify_attestation(self) -> Result<ModelClient<AttestationReceived>> {
        info!("Verifying attestation from server...");
        
        let attestation = self.attestation.as_ref()
            .context("No attestation data received")?;
        
        // TODO CS: verify attestation
        let attestation_str = String::from_utf8_lossy(attestation);
        info!("Attestation data: {}", attestation_str);
        
        Ok(ModelClient {
            stream: self.stream,
            attestation: self.attestation,
            state: PhantomData,
        })
    }
}

impl ModelClient<AttestationReceived> {
    #[instrument(skip(self))]
    pub async fn complete_session(mut self) -> Result<()> {
        info!("Waiting for final completion message from server...");
        
        let stream = self.stream.as_mut()
            .context("Stream not connected")?;
        
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

