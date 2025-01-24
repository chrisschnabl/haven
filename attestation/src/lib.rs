// src/lib.rs

pub mod error;
pub mod validate;

// Re-export the key types so end-users can do: use attestation::{NitroAdDoc, NitroAdError, ...};
pub use error::NitroAdError;
pub use validate::{NitroAdDoc, NitroAdDocPayload};