use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};
use tracing::{instrument, debug};

#[instrument(skip(stream), fields(type_name = std::any::type_name::<T>()))]
pub async fn read_message<T>(stream: &mut VsockStream) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes).await?;
    let msg_len = u32::from_le_bytes(len_bytes) as usize;
    debug!("Reading message of length: {}", msg_len);
    
    let mut data = vec![0u8; msg_len];
    stream.read_exact(&mut data).await?;
    
    let message: T = bincode::deserialize(&data)
        .context("Failed to deserialize message")?;
    
    Ok(message)
}

#[instrument(skip(stream, message), fields(type_name = std::any::type_name::<T>()))]
pub async fn write_message<T>(stream: &mut VsockStream, message: &T) -> Result<()>
where
    T: Serialize,
{
    let data = bincode::serialize(message)
        .context("Failed to serialize message")?;
    
    let len_bytes = (data.len() as u32).to_le_bytes();
    debug!("Writing message of length: {}", data.len());
    stream.write_all(&len_bytes).await?;
    
    stream.write_all(&data).await?;
    
    Ok(())
}