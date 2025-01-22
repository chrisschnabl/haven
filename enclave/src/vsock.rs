use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio_vsock::VsockStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::instrument;

pub const BUFFER_SIZE: usize = 10 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub enum Operation {
    SendFile,
    EofFile,
    Prompt,
    Attestation,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub op: Operation,
    pub data: Vec<u8>,
}

/// Read a `Message` from the stream using a 4-byte length prefix before the bincode payload.
#[instrument(skip(stream))]
pub async fn read_message(stream: &mut VsockStream) -> Result<Message> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let msg_len = u32::from_be_bytes(len_buf) as usize;

    let mut buf = vec![0u8; msg_len];
    stream.read_exact(&mut buf).await?;

    let msg: Message = bincode::deserialize(&buf)?;
    Ok(msg)
}

/// Write a `Message` to the stream using a 4-byte length prefix before the bincode payload.
#[instrument(skip(stream, msg))]
pub async fn write_message(stream: &mut VsockStream, msg: &Message) -> Result<()> {
    let encoded = bincode::serialize(msg)?;
    let len_bytes = (encoded.len() as u32).to_be_bytes();

    stream.write_all(&len_bytes).await?;
    stream.write_all(&encoded).await?;

    Ok(())
}