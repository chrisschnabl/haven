use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Operation {
    SendFile,       
    EofFile,        
    Prompt,         
    EofPrompt,      
    Progress,       
    Attestation,    
    Complete,
}

// TODO CS: refactor messages to not have file pathi ncluded everywhere but clashes iwth bincode 
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub op: Operation,
    pub file_path: Option<String>,
    pub data: Vec<u8>,
}

pub async fn read_message(stream: &mut VsockStream) -> Result<Message> {
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes).await?;
    let msg_len = u32::from_le_bytes(len_bytes) as usize;
    
    let mut data = vec![0u8; msg_len];
    stream.read_exact(&mut data).await?;
    
    let message: Message = bincode::deserialize(&data)
        .context("Failed to deserialize message")?;
    
    Ok(message)
}

pub async fn write_message(stream: &mut VsockStream, message: &Message) -> Result<()> {
    let data = bincode::serialize(message)
        .context("Failed to serialize message")?;
    
    let len_bytes = (data.len() as u32).to_le_bytes();
    stream.write_all(&len_bytes).await?;
    
    stream.write_all(&data).await?;
    
    Ok(())
}