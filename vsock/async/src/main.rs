use clap::{App, Arg};
use futures::StreamExt as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::fs::File;
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};

const BUFFER_SIZE: usize = 65536;


/*
We use this to transfer data between a Host and an Enclave. We do this to avoid baking data into the secure Nitro image. 

Client runs on Host 
Server runs on Enclave

Host
- stores model
- stores datasets

Enclave
- contains llama.cpp
- contains evaluation code 

Start-Up
- Host sends model to enclave
-> How big is it? How long does it take to run?
- Server receives model 
-> Uses llama crate to run model

Next steps:
-> Test transfer of model locally
-> Test hello world exchange on AWS Nitro
-> Test transfer of model on AWS nitro
-> Test inference of model locally
-> Test inference on AWS nitro

Next steps after that:
-> Do the same with datasets
*/

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
    let addr = VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, port);
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