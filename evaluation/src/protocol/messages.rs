use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};

// TODO CS: refactor messages to be better i.e. file path includedi n every method righ now 
// TODO CS: can you dcouple emssages and operations from the read and write

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Operation {
    SendFile,       // Send a chunk of a file
    EofFile,        // End of file marker
    Prompt,         // Send a prompt for evaluation
    EofPrompt,      // End of prompt marker
    Progress,       // Progress update during evaluation
    Attestation,    // Attestation data
    Complete,       // Session complete marker
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub op: Operation,
    pub file_path: Option<String>,
    pub data: Vec<u8>,
}