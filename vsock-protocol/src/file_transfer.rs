use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tracing::info;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use std::path::PathBuf;

use crate::protocol::{read_message, write_message, Operation, Message};


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

pub async fn receive_file(stream: &mut VsockStream, description: &str) -> Result<PathBuf> {
    info!("Waiting for {} file...", description);
    
    let mut file_path = None;
    let mut file: Option<File> = None;
    let mut total_received = 0u64;
    
    loop {
        let msg = read_message(stream).await
            .context(format!("Failed to read message while receiving {}", description))?;
        
        match msg.op {
            Operation::SendFile => {
                if file_path.is_none() && msg.file_path.is_some() {
                    file_path = msg.file_path.clone();
                    info!("Receiving {}: {:?}", description, file_path);
                    
                    if let Some(path) = &file_path {
                        file = Some(File::create(path).await?);
                    }
                }
                
                if let Some(ref mut f) = file {
                    f.write_all(&msg.data).await?;
                    total_received += msg.data.len() as u64;
                    info!("Received {} bytes of {} data", total_received, description);
                }
            }
            
            Operation::EofFile => {
                info!("{} file transfer complete. Received {} bytes", description, total_received);
                break;
            }
            
            _ => {
                return Err(anyhow::anyhow!("Unexpected operation during {} transfer: {:?}", description, msg.op));
            }
        }
    }
    
    if let Some(path) = file_path {
        Ok(PathBuf::from(path))
    } else {
        Err(anyhow::anyhow!("No file path received for {}", description))
    }
}