use anyhow::{Context, Result};
use tokio_vsock::{VsockAddr, VsockStream};
use tracing::{info, error};
use crate::vsock::{write_message, read_message, Message, Operation};
use tokio::fs::File;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
    
        let mut collected = String::new();
        loop {
            let msg = match read_message(&mut stream).await {
                Ok(msg) => {
                    info!("Received message: {:?}", msg);
                    msg
                }
                Err(e) => {
                    error!("Failed to read message: {}", e);
                    anyhow::bail!("Failed to read message: {}", e);
                }
            };
            match msg.op {
                Operation::Prompt => {
                    let token = String::from_utf8(msg.data)?;
                    info!("Received token: {}", token);
                    collected.push_str(&token);
                }
                Operation::EofPrompt => {
                    info!("Token stream completed.");
                    // TODO CS: make sure we trace everything here
                    // Expect attestation here
                    let _ = match read_message(&mut stream).await {
                        Ok(msg) => match msg.op {
                            Operation::Attestation => {
                                info!("Attestation response received");
                                let nonce = hex::decode("0000000000000000000000000000000000000000").expect("decode nonce failed");
                                
                                let document_data = msg.data;
                                let document = parse_document(&document_data).expect("parse document failed");
                                let payload = parse_payload(&document.payload).expect("parse payload failed");
                            
                                match parse_verify_with(document_data, nonce, unix_time) {
                                    Ok((payload, attestation_document)) => {
                                        // TODO CS: check PCRs against expectation
                                        info!("payload {:?}", payload.pcrs);
                                    }
                                    // TODO CS: do not panic here, but bail
                                    Err(e) => panic!("parse_verify_with failed: {:?}", e.to_string()),
                                }
                            
                                println!("user data {:?}", payload.user_data);
                                // TODO CS: verify user data against expectation
                                },
                            _ => anyhow::bail!("Unexpected operation: {:?}", msg.op),
                        },
                        Err(e) => {
                            anyhow::bail!("Failed to read message: {}", e);
                        }
                    };
                    break;
                }
                _ => {
                    anyhow::bail!("Unexpected operation: {:?}", msg.op);
                }
            }
        }    
    }

  
    // TODO CS: handle attestatin here, and check it
    // TODO CS: tihnk about the system picture more 
    // TODO CS: the protocol includes quite some boilerplate, think about how to fix this
    // -- we kind of like have a series of operations, that depend (in terms of time) on each other
    // TODO CS: also think about the throughput of prompt requests, once we do run evaluations on scale

    // TODO CS: what would be nice is to basically do an interactive terminal chat gpt mode
    Ok(())
}
