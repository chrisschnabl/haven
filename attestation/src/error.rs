// src/error.rs

use thiserror::Error;
// NOTE: In aws-nitro-enclaves-cose 0.5, the error is named `CoseError` not `COSEError`.
use aws_nitro_enclaves_cose::error::CoseError;
use serde_cbor;
use serde_json;

/// A custom error type for your attestation logic.
#[derive(Debug, Error)]
pub enum NitroAdError {
    #[error("COSE Error: {0}")]
    CoseError(#[from] CoseError),

    #[error("CBOR Deserialization Error: {0}")]
    CborError(#[from] serde_cbor::Error),

    #[error("JSON Error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("General error: {0}")]
    GenericError(String),
}

// Optional: implement From<webpki::Error> if you want to use `?` with webpki
/*
impl From<webpki::Error> for NitroAdError {
    fn from(e: webpki::Error) -> Self {
        NitroAdError::GenericError(format!("webpki error: {:?}", e))
    }
}
*/
