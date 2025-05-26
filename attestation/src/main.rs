use std::fs;

use tee_attestation_verifier::{parse_verify_with, parse_document, parse_payload};

fn main() {
    let unix_time = std::time::UNIX_EPOCH.elapsed().unwrap().as_secs();
    let attestation_doc_path = "attestation_response.txt";
    let document_data = fs::read(attestation_doc_path)
        .expect("Failed to read attestation_response.txt");

    let nonce =
        hex::decode("0000000000000000000000000000000000000000").expect("decode nonce failed");

    let document = parse_document(&document_data).expect("parse document failed");
    let payload = parse_payload(&document.payload).expect("parse payload failed");

    match parse_verify_with(document_data, nonce, unix_time) {
        Ok((payload, attestation_document)) => {
            println!("payload {:?}, attestation_document {:?}", payload.pcrs, attestation_document);
        }
        Err(e) => panic!("parse_verify_with failed: {:?}", e.to_string()),
    }

    println!("user data {:?}", payload.user_data);
}
