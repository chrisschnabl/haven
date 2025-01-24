use std::fs;
use attestation::NitroAdDoc; // or whatever your crate is called
use openssl::x509::X509;

fn main() {
    // 1. Load PEM-encoded root cert
    let root_cert_pem = fs::read("root.pem")
        .expect("Failed to read root.pem");
    
    // 2. Convert PEM -> DER with OpenSSL
    let root_cert = X509::from_pem(&root_cert_pem)
        .expect("Could not parse PEM file as X.509 certificate");
    let root_cert_der = root_cert.to_der()
        .expect("Failed to convert PEM certificate to DER");

    // 3. Load the attestation response (COSE) bytes
    let attestation_bytes = fs::read("attestation_response.txt")
        .expect("Failed to read attestation_response.txt");
    
    let unix_ts_sec = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();
        
    let fake_past_ts = 1737661663u64;
    let attestation_doc = match NitroAdDoc::from_bytes(
        &attestation_bytes,
        &root_cert_der,
        fake_past_ts
    ) {
        Ok(doc) => doc,
        Err(e) => {
            eprintln!("Error verifying attestation doc: {:#?}", e);
            std::process::exit(1);
        }
    };

    // 5. Print the resulting JSON if everything is valid
    match attestation_doc.to_json() {
        Ok(json_output) => {
            println!("Attestation document (JSON):\n{json_output}");
        }
        Err(e) => {
            eprintln!("Error converting attestation doc to JSON: {e}");
            std::process::exit(1);
        }
    }
}
