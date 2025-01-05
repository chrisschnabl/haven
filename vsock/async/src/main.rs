use clap::{App, Arg};
use futures::StreamExt as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_vsock::{VsockAddr, VsockListener, VsockStream};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("test_program")
        .version("1.0")
        .author("Chris Schnabl <chris.schnabl.cs@gmail.com>")
        .about("Tokio Virtio socket test program (server and client)")
        .arg(
            Arg::with_name("mode")
                .long("mode")
                .short("m")
                .help("Mode of operation: server or client")
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
        .get_matches();

    let mode = matches.value_of("mode").expect("Mode is required");
    let port = matches
        .value_of("port")
        .expect("Port is required")
        .parse::<u32>()
        .expect("Port must be a valid integer");

    if mode == "server" {
        run_server(port).await?;
    } else if mode == "client" {
        run_client(port).await?;
    } else {
        eprintln!("Invalid mode. Use 'server' or 'client'.");
        std::process::exit(1);
    }

    Ok(())
}

async fn run_server(port: u32) -> Result<(), Box<dyn std::error::Error>> {
    let addr = VsockAddr::new(libc::VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr).expect("Unable to bind Virtio listener");

    println!("Listening for connections on port: {}", port);

    let mut incoming = listener.incoming();
    while let Some(result) = incoming.next().await {
        match result {
            Ok(mut stream) => {
                println!("Got connection ============");
                tokio::spawn(async move {
                    loop {
                        let mut buf = vec![0u8; 5000];
                        let len = stream.read(&mut buf).await.unwrap();

                        if len == 0 {
                            break;
                        }

                        buf.resize(len, 0);
                        println!("Got data: {:?}", &buf);
                        stream.write_all(&buf).await.unwrap();
                    }
                });
            }
            Err(e) => {
                println!("Got error: {:?}", e);
            }
        }
    }

    Ok(())
}

async fn run_client(port: u32) -> Result<(), Box<dyn std::error::Error>> {
    let addr = VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, port);
    let mut stream = VsockStream::connect(addr)
        .await
        .expect("Failed to connect to server");

    println!(
        "Connected to server at CID {}:PORT {}",
        tokio_vsock::VMADDR_CID_LOCAL,
        port
    );

    let message = b"Hello from client!";
    stream
        .write_all(message)
        .await
        .expect("Failed to send data");

    println!("Sent: {:?}", message);

    // Receive the echoed response from the server
    let mut buf = vec![0u8; 5000];
    let len = stream.read(&mut buf).await.expect("Failed to read data");

    if len > 0 {
        buf.resize(len, 0);
        println!("Received: {:?}", &buf);
    } else {
        println!("No data received from server");
    }

    Ok(())
}
