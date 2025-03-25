use anyhow::{Context, Result};
use tokio_vsock::VsockStream;
use tracing::info;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use std::path::{Path, PathBuf};
use tokio::io::AsyncReadExt;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use std::time::Duration;
use crate::messages::{write_message, read_message, Operation, Message};


const BUFFER_SIZE: usize = 10 * 1024 * 1024;

pub async fn send_file(stream: &mut VsockStream, file_path: &str) -> Result<()> {
    info!("Sending file: {}", file_path);
    
    let mut file = File::open(file_path).await
        .context(format!("Failed to open file: {}", file_path))?;
    
    let mut buffer = vec![0; BUFFER_SIZE];
    
    let path = Path::new(file_path);
    let file_name = path.file_name().unwrap().to_str().unwrap();

    let mut msg = Message {
        op: Operation::SendFile,
        file_path: Some(file_name.to_string()),
        data: Vec::with_capacity(BUFFER_SIZE),
    };
    write_message(stream, &msg).await?;
    
    let file_size = file.metadata().await?.len();
    let mut bytes_sent = 0;

    let pb = ProgressBar::new(file_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Sending '{}'", file_path));
        
    loop {
        let n = file.read(&mut buffer).await?;
        if n == 0 {
            break;
        }
        
        msg.data.clear();
        msg.data.extend_from_slice(&buffer[..n]);
        write_message(stream, &msg).await?;
        
        bytes_sent += n as u64;
        pb.set_position(bytes_sent);
    }
    
    msg.op = Operation::EofFile;
    msg.data.clear();
    write_message(stream, &msg).await?;
    
    pb.finish_with_message(format!("Sent '{}' ({} bytes)", file_path, bytes_sent));
    info!("File transfer complete: {} ({} bytes sent)", file_path, bytes_sent);
    Ok(())
}

pub async fn receive_file(stream: &mut VsockStream, description: &str) -> Result<PathBuf> {
    info!("Waiting for {} file...", description);
    
    let mut file_path = None;
    let mut file: Option<File> = None;
    let mut total_received = 0u64;

    let pb = ProgressBar::new_spinner();
    pb.set_message("Receiving file...");
    pb.enable_steady_tick(Duration::from_millis(100));
    
    loop {
        let msg = read_message(stream).await
            .context(format!("Failed to read message while receiving {}", description))?;
        
        match msg.op {
            Operation::SendFile => {
                if file_path.is_none() && msg.file_path.is_some() {
                    file_path = msg.file_path.clone();
                    info!("Receiving {}: {:?}", description, file_path);
                    
                    // if start with dir, create it
                    if let Some(path) = &file_path {
                        file = Some(File::create(path).await?);
                    }
                }

                if let Some(ref mut f) = file {
                    f.write_all(&msg.data).await?;
                    total_received += msg.data.len() as u64;
                    pb.set_message(format!("Received {} bytes of {} data", total_received, description));
                }
            }
            
            Operation::EofFile => {
                pb.finish_with_message("File transfer complete. Received EOF marker.");
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