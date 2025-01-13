use clap::{Parser, ValueEnum};
use futures::StreamExt as _;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::fs::File;
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::time::Duration;
use std::convert::TryInto;
use std::io::Write;
use serde::{Serialize, Deserialize};
use indicatif::{ProgressBar, ProgressStyle};
// TOOD CS: reorganize

const BUFFER_SIZE: usize = 10 * 1024 * 1024; // 10 MB

#[derive(Debug, Serialize, Deserialize)]
enum Operation {
    SendFile,
    EofFile,
}

#[derive(Debug, Serialize, Deserialize)]
struct Message {
    op: Operation,
    data: Vec<u8>,
}

// We'll use a 4-byte length prefix before the bincode payload.
async fn read_message(stream: &mut VsockStream) -> Result<Message> {
    // 1. Read the 4-byte length
    let mut len_buf = [0u8; 4];
    let n = stream.read_exact(&mut len_buf).await?;
    if n == 0 {
        bail!("EOF reached (no length bytes)");
    }
    let msg_len = u32::from_be_bytes(len_buf) as usize;

    // 2. Read the payload
    let mut buf = vec![0u8; msg_len];
    stream.read_exact(&mut buf).await?;

    // 3. Deserialize
    let msg: Message = bincode::deserialize(&buf)?;
    Ok(msg)
}

async fn write_message(stream: &mut VsockStream, msg: &Message) -> Result<()> {
    let encoded = bincode::serialize(msg)?;

    let len_bytes = (encoded.len() as u32).to_be_bytes();
    tokio::io::AsyncWriteExt::write_all(stream, &len_bytes).await?;
    tokio::io::AsyncWriteExt::write_all(stream, &encoded).await?;

    Ok(())
}

// ---------------------------------------------------------------------

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
    #[arg(long, short, required_if_eq("mode", "host"))]
    file: Option<String>,

    /// Alternative option to specify data directly (required in 'host' mode if file is not provided)
    #[arg(long, default_value = "This is the default prompt. Please tell a story.")]
    prompt: String,

    #[arg(long, short, default_value_t = libc::VMADDR_CID_ANY)]
    cid: u32,
}

#[derive(ValueEnum, Clone)]
enum Mode {
    Enclave,
    Host,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    match args.mode {
        Mode::Enclave => {
            println!("Starting in enclave (server) mode...");
            run_server(args.port).await?;
        }
        Mode::Host => {
            let file_path = args
                .file
                .expect("File path is required in 'host' mode");
            println!("Starting in host (client) mode...");
            run_client(args.port, args.cid, &file_path, &args.prompt).await?;
        }
    }

    Ok(())
}

async fn run_server(port: u32) -> Result<()> {
    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr).expect("Unable to bind Virtio listener");

    println!("Listening for connections on port: {}", port);

    let mut incoming = listener.incoming();
    while let Some(result) = incoming.next().await {
        match result {
            Ok(mut stream) => {
                println!("Got connection ============");
                tokio::spawn(async move {
                    if let Err(e) = handle_incoming_messages(&mut stream).await {
                        eprintln!("Error handling client: {:?}", e);
                    }
                });
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }

    Ok(())
}

/// Receives `Message` structs from the client. If `op == SendFile`, writes the data
/// to `model.gguf`. If `op == EofFile`, stops receiving and calls `test_model()`.
async fn handle_incoming_messages(stream: &mut VsockStream) -> Result<()> {
    let mut file = File::create("model.gguf")
        .await
        .expect("Failed to create file 'model.gguf'");

    loop {
        let msg = match read_message(stream).await {
            Ok(m) => m,
            Err(e) => {
                // If we can't read a message properly, let's just break out.
                eprintln!("Error reading bincode message: {:?}", e);
                break;
            }
        };

        let mut total_received = 0;
        let pb = ProgressBar::new_spinner();
        pb.set_message("Receiving file...");
        pb.enable_steady_tick(Duration::from_millis(100));

        match msg.op {
            Operation::SendFile => {
                // Write the incoming bytes to disk
                file.write_all(&msg.data).await?;
                total_received += msg.data.len() as u64;
                pb.set_message(format!("Wrote {} bytes to 'model.gguf'", total_received));
            }
            Operation::EofFile => {
                pb.finish_with_message("File transfer complete. Received EOF marker.");
                println!("Received end-of-file marker. Stopping file reception.");
                // Once done, we can call `test_model()`
                println!("File transfer complete. Now let's test the model...");
                if let Err(e) = test_model() {
                    eprintln!("Error running test_model: {:?}", e);
                }
                break;
            }
        }
    }
    Ok(())
}

async fn run_client(port: u32, cid: u32, file_path: &str, prompt: &str) -> Result<()> {
    let addr = VsockAddr::new(cid, port);

    let mut stream = VsockStream::connect(addr)
        .await
        .context("Failed to connect to server")?;

    println!(
        "Connected to server at CID {}:PORT {}",
        tokio_vsock::VMADDR_CID_LOCAL,
        port
    );

    // Send the file in chunks as Message
    let mut file = File::open(file_path)
        .await
        .context("Failed to open file for reading")?;

    let mut buf = vec![0u8; BUFFER_SIZE];
    let file_metadata = tokio::fs::metadata(file_path).await?;
    let total_size = file_metadata.len();
    let mut total_sent = 0;

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("Failed to set template for progress bar")
            .progress_chars("#>-")
    );

    loop {
        let len = file.read(&mut buf).await?;
        if len == 0 {
            // EOF reached => send EofFile
            let msg = Message {
                op: Operation::EofFile,
                data: vec![],
            };
            write_message(&mut stream, &msg).await?;
            pb.finish_with_message("File transfer complete. Sent EOF marker.");
            break;
        }

        // Create a Message to hold this chunk
        let msg = Message {
            op: Operation::SendFile,
            data: buf[..len].to_vec(),
        };
        // Send it
        write_message(&mut stream, &msg).await?;
        total_sent += len as u64;
        pb.set_position(total_sent);
    }

    Ok(())
}

fn test_model() -> Result<()> {
    let model_path = "model.gguf";
    let prompt = "Hello, this is a test prompt.";
    let n_len = 32;
    let seed = 1337;
    let threads = 2;
    let ctx_size = NonZeroU32::new(2048).unwrap();

    let backend = LlamaBackend::init().context("Failed to initialize LlamaBackend")?;
    let model_params = LlamaModelParams::default();

    println!("Loading model from path: {:?}", model_path);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .context("Unable to load model")?;
    println!("Model loaded successfully!");

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_threads(threads);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("Unable to create the llama_context")?;

    // Tokenize the prompt
    let tokens_list = model
        .str_to_token(prompt, AddBos::Always)
        .context(format!("Failed to tokenize {prompt}"))?;

    let n_ctx = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);
    if n_kv_req > n_ctx {
        bail!(
            "n_kv_req > n_ctx: the required KV cache size is not big enough. \
            Either reduce n_len or increase n_ctx."
        );
    }
    if tokens_list.len() >= n_len.try_into()? {
        bail!("The prompt is too long; it has more tokens than n_len.");
    }

    // Print the prompt token-by-token
    for token in &tokens_list {
        print!("{}", model.token_to_str(*token, Special::Tokenize)?);
    }
    std::io::stdout().flush()?;

    // Create a batch for decoding
    let mut batch = LlamaBatch::new(512, 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .context("llama_decode() failed")?;

    // Main loop for token generation
    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;
    let t_main_start = ggml_time_us();
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(seed),
        LlamaSampler::greedy(),
    ]);
    while n_cur <= n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            println!();
            break;
        }

        let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let output_string = String::from_utf8(output_bytes)?;
        print!("{output_string}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;

        n_cur += 1;

        ctx.decode(&mut batch)
            .context("Failed to eval")?;

        n_decode += 1;
    }

    println!();
    let t_main_end = ggml_time_us();

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

    println!(
        "Decoded {} tokens in {:.2} s, speed {:.2} t/s",
        n_decode,
        duration.as_secs_f32(),
        n_decode as f32 / duration.as_secs_f32()
    );

    println!("{}", ctx.timings());

    Ok(())
}