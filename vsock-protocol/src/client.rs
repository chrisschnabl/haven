use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockStream};
use tracing::{info, instrument};
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use std::path::Path;

use crate::protocol::{write_message, Message, Operation};

/// Constant for file transfer buffer size
pub const BUFFER_SIZE: usize = 4096;

/// Connects to a vsock server
#[instrument]
pub async fn connect_to_server(cid: u32, port: u32) -> Result<VsockStream> {
    info!("Connecting to server at CID {}, port {}", cid, port);
    
    let addr = VsockAddr::new(cid, port);
    let stream = VsockStream::connect(addr)
        .await
        .context("Failed to connect to server")?;
    
    info!("Successfully connected to server");
    Ok(stream)
}

/// Sends a file over the vsock connection
#[instrument(skip(stream))]
pub async fn send_file(stream: &mut VsockStream, file_path: &str) -> Result<()> {
    let path = Path::new(file_path);
    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path);
    
    info!("Sending file: {}", file_name);
    
    let mut file = File::open(path)
        .await
        .context(format!("Failed to open file: {}", file_path))?;
    
    let file_metadata = tokio::fs::metadata(path).await?;
    let total_size = file_metadata.len();
    let mut total_sent = 0;
    
    let mut buf = vec![0u8; BUFFER_SIZE];

    loop {
        let len = file.read(&mut buf).await?;
        if len == 0 {
            let msg = Message {
                op: Operation::EofFile,
                file_path: Some(file_path.to_string()),
                data: vec![],
            };
            write_message(stream, &msg).await?;
            info!("File transfer complete. Sent EOF marker.");
            break;
        }

        let msg = Message {
            op: Operation::SendFile,
            file_path: Some(file_path.to_string()),
            data: buf[..len].to_vec(),
        };
        write_message(stream, &msg).await?;

        total_sent += len as u64;
        info!("Sent {} of {} bytes", total_sent, total_size);
    }
    
    Ok(())
}