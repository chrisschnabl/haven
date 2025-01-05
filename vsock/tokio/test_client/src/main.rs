use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_vsock::VsockStream;
use clap::{App, Arg};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("test_client")
        .version("1.0")
        .author("Your Name <your_email@example.com>")
        .about("Tokio Virtio socket test client")
        /*.arg(
            Arg::with_name("cid")
                .long("cid")
                .short("c")
                .help("CID of the server (typically VMADDR_CID_LOCAL for local VM)")
                .required(true)
                .takes_value(true),
        )*/
        .arg(
            Arg::with_name("port")
                .long("port")
                .short("p")
                .help("Port of the server to connect to")
                .required(true)
                .takes_value(true),
        )
        .get_matches();

    /*let server_cid = matches
        .value_of("cid")
        .expect("CID is required")
        .parse::<u32>()
        .expect("CID must be a valid integer");*/

    let server_port = matches
        .value_of("port")
        .expect("Port is required")
        .parse::<u32>()
        .expect("Port must be a valid integer");

    // Connect to the server
    let addr = tokio_vsock::VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, server_port);
    let mut stream = VsockStream::connect(addr).await.expect("Failed to connect to server");

    println!("Connected to server at CID {}:PORT {}", tokio_vsock::VMADDR_CID_LOCAL, server_port);

    // Send a message to the server
    let message = b"Hello from client!";
    stream.write_all(message).await.expect("Failed to send data");
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
