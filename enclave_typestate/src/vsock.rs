// src/vsock.rs

use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};

// Buffer size for chunk transfers
pub const BUFFER_SIZE: usize = 4096;

// Operation types for the protocol
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Operation {
    // File operations
    SendFile,       // Send a chunk of a file
    EofFile,        // End of file marker
    
    // Evaluation operations
    Prompt,         // Send a prompt for evaluation
    EofPrompt,      // End of prompt marker
    Progress,       // Progress update during evaluation
    
    // Attestation operations
    Attestation,    // Attestation data
    
    // Session management
    Complete,       // Session complete marker
}

// Message structure for the protocol
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub op: Operation,
    pub file_path: Option<String>,
    pub data: Vec<u8>,
}

// Read a message from the given stream
pub async fn read_message(stream: &mut VsockStream) -> Result<Message> {
    // Read message length (u32)
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes).await?;
    let msg_len = u32::from_le_bytes(len_bytes) as usize;
    
    // Read message data
    let mut data = vec![0u8; msg_len];
    stream.read_exact(&mut data).await?;
    
    // Deserialize the message
    let message: Message = bincode::deserialize(&data)
        .context("Failed to deserialize message")?;
    
    Ok(message)
}

// Write a message to the given stream
pub async fn write_message(stream: &mut VsockStream, message: &Message) -> Result<()> {
    // Serialize the message
    let data = bincode::serialize(message)
        .context("Failed to serialize message")?;
    
    // Write message length
    let len_bytes = (data.len() as u32).to_le_bytes();
    stream.write_all(&len_bytes).await?;
    
    // Write message data
    stream.write_all(&data).await?;
    
    Ok(())
}