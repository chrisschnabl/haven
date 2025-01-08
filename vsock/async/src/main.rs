use clap::{App, Arg};
use futures::StreamExt as _;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::fs::File;
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};
//use llama_cpp::{LlamaModel, LlamaParams, SessionParams, standard_sampler::StandardSampler};

const BUFFER_SIZE: usize = 65536;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("file_transfer_program")
        .version("1.0")
        .author("Chris Schnabl <chris.schnabl.cs@gmail.com>")
        .about("Tokio Virtio socket file transfer program (enclave and host)")
        .arg(
            Arg::with_name("mode")
                .long("mode")
                .short("m")
                .help("Mode of operation: 'enclave' (server) or 'host' (client)")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("port")
                .long("port")
                .short("p")
                .help("Port for the server or client")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("file")
                .long("file")
                .short("f")
                .help("Path to the file to send (required in 'host' mode)")
                .required_if("mode", "host")
                .takes_value(true),
        )
        .get_matches();

    let mode = matches.value_of("mode").expect("Mode is required");
    let port = matches
        .value_of("port")
        .expect("Port is required")
        .parse::<u32>()
        .expect("Port must be a valid integer");

    if mode == "enclave" {
        println!("Starting in enclave (server) mode...");
        run_server(port).await?;
        //test_llama().await?;
    } else if mode == "host" {
        let file_path = matches
            .value_of("file")
            .expect("File path is required in 'host' mode");
        println!("Starting in host (client) mode...");
        run_client(port, file_path).await?;
    } else {
        eprintln!("Invalid mode. Use 'enclave' or 'host'.");
        std::process::exit(1);
    }

    Ok(())
}

/*
async fn test_llama() -> Result<(), Box<dyn std::error::Error>> {
    let model = LlamaModel::load_from_file("path_to_model.gguf", LlamaParams::default())?;
    let mut ctx = model.create_session(SessionParams::default())?;

    ctx.advance_context("This is the story of a man named Stanley.")?;

    let max_tokens = 1024;
    let mut decoded_tokens = 0;

    /*let mut completions = match ctx.start_completing_with(StandardSampler::default(), 1024) {
        Ok(handle) => handle.into_strings(),
        Err(e) => {
            eprintln!("Error: {:?}", e);
            return Err(e.into());
        }
    };*/

    let mut completions = ctx.start_completing_with(StandardSampler::default(), max_tokens)?
        .into_strings();

    let mut stdout = tokio::io::stdout(); // Use Tokio's asynchronous stdout

    while let Some(completion) = futures::StreamExt::next(&mut completions).await {
        stdout.write_all(completion.as_bytes()).await?;
        stdout.flush().await?;

        decoded_tokens += 1;

        if decoded_tokens >= max_tokens {
            break;
        }
    }

    Ok(())
}*/


async fn run_server(port: u32) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr).expect("Unable to bind Virtio listener");

    println!("Listening for connections on port: {}", port);

    let mut incoming = listener.incoming();
    while let Some(result) = incoming.next().await {
        match result {
            Ok(mut stream) => {
                println!("Got connection ============");
                tokio::spawn(async move {
                    let mut file = File::create("received_file.txt")
                        .await
                        .expect("Failed to create file");

                    loop {
                        let mut buf = vec![0u8; BUFFER_SIZE];
                        let len = match stream.read(&mut buf).await {
                            Ok(len) if len > 0 => len,
                            _ => break, // EOF or connection closed
                        };

                        buf.resize(len, 0);

                        if let Err(e) = file.write_all(&buf).await {
                            eprintln!("Failed to write to file: {:?}", e);
                            break;
                        }

                        println!("Wrote {} bytes to file", len);
                    }

                    println!("File transfer complete");
                });
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }

    Ok(())
}

async fn run_client(port: u32, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // let addr = VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, port);
    let addr = VsockAddr::new(16, port);
    let mut stream = VsockStream::connect(addr)
        .await
        .expect("Failed to connect to server");

    println!(
        "Connected to server at CID {}:PORT {}",
        tokio_vsock::VMADDR_CID_LOCAL,
        port
    );

    let mut file = File::open(file_path)
        .await
        .expect("Failed to open file for reading");

    let mut buf = vec![0u8; BUFFER_SIZE]; // 64 KB buffer
    loop {
        let len = file.read(&mut buf).await?;
        if len == 0 {
            break; // EOF reached
        }

        stream
            .write_all(&buf[..len])
            .await
            .expect("Failed to send data");

        println!("Sent {} bytes to server", len);
    }

    println!("File transfer complete");

    Ok(())
}