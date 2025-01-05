use rand::RngCore;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(target_os = "linux")]
use tokio_vsock::VsockListener;
use tokio_vsock::{VsockAddr, VsockStream};

const TEST_BLOB_SIZE: usize = 100_000;
const TEST_BLOCK_SIZE: usize = 5_000;

#[tokio::test]
async fn test_vsock_server() {
    let mut rng = rand::thread_rng();
    let mut blob: Vec<u8> = vec![];
    let mut rx_blob = vec![];
    let mut tx_pos = 0;

    blob.resize(TEST_BLOB_SIZE, 0);
    rx_blob.resize(TEST_BLOB_SIZE, 0);
    rng.fill_bytes(&mut blob);

    // Use VMADDR_CID_LOCAL for local loopback testing
    let addr = VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, 8000);
    let listener = VsockListener::bind(addr).expect("Failed to bind vsock listener");

    // Spawn a server task
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("Failed to accept connection");
        let mut buffer = [0u8; TEST_BLOCK_SIZE];

        loop {
            let n = stream.read(&mut buffer).await.expect("Failed to read data");
            if n == 0 {
                break; // Connection closed
            }
            stream.write_all(&buffer[..n]).await.expect("Failed to write data");
        }
    });

    let mut stream = VsockStream::connect(addr).await.expect("Connection failed");

    while tx_pos < TEST_BLOB_SIZE {
        let written_bytes = stream
            .write(&blob[tx_pos..tx_pos + TEST_BLOCK_SIZE])
            .await
            .expect("Write failed");
        if written_bytes == 0 {
            panic!("Stream unexpectedly closed");
        }

        let mut rx_pos = tx_pos;
        while rx_pos < (tx_pos + written_bytes) {
            let read_bytes = stream
                .read(&mut rx_blob[rx_pos..])
                .await
                .expect("Read failed");
            if read_bytes == 0 {
                panic!("Stream unexpectedly closed");
            }
            rx_pos += read_bytes;
        }

        tx_pos += written_bytes;
    }

    let expected = Sha256::digest(&blob);
    let actual = Sha256::digest(&rx_blob);

    assert_eq!(expected, actual);
}

#[tokio::test]
async fn test_vsock_conn_error() {
    let addr = VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, 8001); // Unused port
    let err = VsockStream::connect(addr)
        .await
        .expect_err("Connection succeeded")
        .raw_os_error()
        .expect("Not an OS error");

    if err == 0 {
        panic!("Non-zero error expected");
    }
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn split_vsock() {
    const MSG: &[u8] = b"split";
    const PORT: u32 = 8002;

    let addr = VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, PORT);
    let listener = VsockListener::bind(addr).expect("Failed to bind listener");

    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener
            .accept()
            .await
            .expect("Failed to accept connection");
        stream.write_all(MSG).await.expect("Failed to write");
        let mut read_buf = [0u8; 32];
        let read_len = stream.read(&mut read_buf).await.expect("Failed to read");
        assert_eq!(&read_buf[..read_len], MSG);
    });

    let mut stream = VsockStream::connect(addr).await.expect("Connection failed");
    let (mut read_half, mut write_half) = stream.split();

    let mut read_buf = [0u8; 32];
    let read_len = read_half
        .read(&mut read_buf)
        .await
        .expect("Failed to read from vsock");
    assert_eq!(&read_buf[..read_len], MSG);

    assert_eq!(
        write_half
            .write(MSG)
            .await
            .expect("Failed to write to vsock"),
        MSG.len()
    );

    handle.await.expect("Task failed");
}

#[tokio::test]
#[cfg(target_os = "linux")]
async fn into_split_vsock() {
    const MSG: &[u8] = b"split";
    const PORT: u32 = 8003;

    let addr = VsockAddr::new(tokio_vsock::VMADDR_CID_LOCAL, PORT);
    let listener = VsockListener::bind(addr).expect("Failed to bind listener");

    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener
            .accept()
            .await
            .expect("Failed to accept connection");
        stream.write_all(MSG).await.expect("Failed to write");
        let mut read_buf = [0u8; 32];
        let read_len = stream.read(&mut read_buf).await.expect("Failed to read");
        assert_eq!(&read_buf[..read_len], MSG);
    });

    let stream = VsockStream::connect(addr).await.expect("Connection failed");
    let (mut read_half, mut write_half) = stream.into_split();

    let mut read_buf = [0u8; 32];
    let read_len = read_half
        .read(&mut read_buf[..])
        .await
        .expect("Failed to read from vsock");
    assert_eq!(&read_buf[..read_len], MSG);

    assert_eq!(
        write_half
            .write(MSG)
            .await
            .expect("Failed to write to vsock"),
        MSG.len()
    );

    handle.await.expect("Task failed");

    // Assert that the halves can be merged together again
    let _ = read_half.unsplit(write_half);
}