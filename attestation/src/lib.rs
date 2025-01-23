use anyhow::Result;
use nsm_io::{Request as NsmRequest, Response as NsmResponse};
use serde_bytes::ByteBuf;
use sha2::{Digest, Sha384};

pub fn generate_attestation(
    input_prompt: &str,
    output_prompt: &str,
    model_id: &str,
) -> Result<Vec<u8>> {
    let input_hash = Sha384::digest(input_prompt.as_bytes());
    let output_hash = Sha384::digest(output_prompt.as_bytes());
    let model_hash = Sha384::digest(model_id.as_bytes());

    let mut user_data = Vec::new();
    user_data.extend_from_slice(&input_hash);
    user_data.extend_from_slice(&output_hash);
    user_data.extend_from_slice(&model_hash);

    let nsm_fd = nsm_driver::nsm_init();
    if nsm_fd < 0 {
        return Err(anyhow::anyhow!("Failed to initialize NSM"));
    }

    let request = NsmRequest::Attestation {
        public_key: None,
        user_data: Some(ByteBuf::from(user_data)),
        nonce: None,
    };

    let response = nsm_driver::nsm_process_request(nsm_fd, request);

    let result = match response {
        NsmResponse::Attestation { document, .. } => Ok(document),
        NsmResponse::Error(error_code) => Err(anyhow::anyhow!("NSM Error: {:?}", error_code)),
        _ => Err(anyhow::anyhow!("Unexpected NSM response")),
    };

    nsm_driver::nsm_exit(nsm_fd);

    result
}