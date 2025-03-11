use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tracing::info;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use std::path::{Path, PathBuf};
use tokio::io::AsyncReadExt;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;

use crate::messages::{write_message, read_message, Operation, Message};


const BUFFER_SIZE: usize = 4096;


pub async fn open_file(path: &Path) -> Result<File> {
    // TODO CS: outside this we often use the path nad not name
    let _ = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path.to_str().unwrap());

    let file = File::open(path)
        .await
        .context(format!("Failed to open file: {}", path.to_str().unwrap()))?;

    Ok(file)
}

pub async fn send_file(stream: &mut VsockStream, file_path: &str) -> Result<()> {
    let path = Path::new(file_path);
    let mut file = open_file(path).await?;
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
    pb.set_message(format!("Transferring '{}'...", file_path));   

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