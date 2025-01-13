use anyhow::Result;
use clap::{Parser, ValueEnum};

/// Import all vsock server/client logic from vsock.rs
mod vsock;
use vsock::{run_client, run_server};

#[derive(Parser)]
#[command(name = "file_transfer_program")]
#[command(version = "0.0.1")]
#[command(author = "Chris Schnabl <chris.schnabl.cs@gmail.com>")]
#[command(about = "Tokio Virtio socket file transfer program (enclave and host)")]
struct Cli {
    /// Mode of operation: 'enclave' (server) or 'host' (client)
    #[arg(long, short)]
    mode: Mode,

    /// Port for the server or client
    #[arg(long, short)]
    port: u32,

    /// Path to the file to send (required in 'host' mode)
    #[arg(long, short)]
    file: Option<String>,

    /// Alternative option to specify data directly (required in 'host' mode if file is not provided)
    #[arg(long, default_value = "Default prompt.")]
    prompt: Option<String>,

    /// CID (defaults to ANY for the server, but you can specify for the client)
    #[arg(long, short, default_value_t = libc::VMADDR_CID_ANY)]
    cid: u32,
}

#[derive(ValueEnum, Clone)]
enum Mode {
    Enclave,
    Host,
}

use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    tracing_subscriber::fmt()
        .init();

    match args.mode {
        Mode::Enclave => {
            println!("Starting in enclave (server) mode...");
            run_server(args.port).await?;
        }
        Mode::Host => {
            println!("Starting in host (client) mode...");
            run_client(args.port, args.cid, args.file.as_deref(), args.prompt.as_deref()).await?;
        }
    }

    Ok(())
}
