use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockStream};
use tracing::{info};
use crate::vsock::{write_message, Message, Operation};
use tokio::fs::File;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::{AsyncReadExt};

pub async fn run_client(port: u32, cid: u32, file_path: Option<&str>, prompt: Option<&str>) -> Result<()> {
    let addr = VsockAddr::new(cid, port);
    let mut stream = VsockStream::connect(addr)
        .await
        .context("Failed to connect to server")?;

    info!(
        "Connected to server at CID {}:PORT {}",
        tokio_vsock::VMADDR_CID_LOCAL,
        port
    );

    if let Some(file_path) = file_path {
        let mut file = File::open(file_path)
            .await
            .context("Failed to open file for reading")?;
    
        let file_metadata = tokio::fs::metadata(file_path).await?;
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

        let mut buf = vec![0u8; crate::vsock::BUFFER_SIZE];

        loop {
            let len = file.read(&mut buf).await?;
            if len == 0 {
                // EOF => send EofFile
                let msg = Message {
                    op: Operation::EofFile,
                    data: vec![],
                };
                write_message(&mut stream, &msg).await?;
                pb.finish_with_message("File transfer complete. Sent EOF marker.");
                break;
            }

            let msg = Message {
                op: Operation::SendFile,
                data: buf[..len].to_vec(),
            };
            write_message(&mut stream, &msg).await?;

            total_sent += len as u64;
            pb.set_position(total_sent);
        }
    }

    if let Some(prompt) = prompt {
        let msg = Message {
            op: Operation::Prompt,
            data: prompt.as_bytes().to_vec(),
        };
        write_message(&mut stream, &msg).await?;
    }
    

    Ok(())
}
