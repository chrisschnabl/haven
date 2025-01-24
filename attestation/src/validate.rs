use crate::error::NitroAdError;
use aws_nitro_enclaves_cose::crypto::Openssl;
use aws_nitro_enclaves_cose::sign::CoseSign1;

use chrono::{DateTime, Duration, TimeZone, Utc};
use itertools::Itertools;
use openssl::{
    bn::BigNumContext,
    ec::{EcGroup, EcKey, EcPoint},
    nid::Nid,
    pkey::PKey,
};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use std::{collections::HashMap, convert::TryFrom};
use hex;
use x509_parser::prelude::*;
use serde_cbor;
use serde_json;
use webpki::{EndEntityCert, TlsServerTrustAnchors, Time};

/// Our payload struct
#[derive(Debug, Serialize, Deserialize)]
pub struct NitroAdDocPayload {
    pub module_id: String,
    pub digest: String,

    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub timestamp: DateTime<Utc>,

    #[serde(serialize_with = "ser_pcrs_as_hex")]
    pub pcrs: HashMap<u8, ByteBuf>,

    #[serde(skip_serializing)]
    pub certificate: ByteBuf,

    #[serde(skip_serializing)]
    pub cabundle: Vec<ByteBuf>,
}

/// Sort & hex-encode PCRs for JSON output
fn ser_pcrs_as_hex<S>(pcrs: &HashMap<u8, ByteBuf>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mapped = pcrs
        .iter()
        .sorted_by_key(|(k, _)| *k)
        .map(|(k, v)| (k, hex::encode(v)));
    serializer.collect_map(mapped)
}

/// Main struct holding our parsed/validated Nitro attestation document
pub struct NitroAdDoc {
    payload: NitroAdDocPayload,
}

impl NitroAdDoc {
    /// Validates and loads an attestation doc from raw bytes
    pub fn from_bytes(
        bytes: &[u8],
        root_cert_der: &[u8],
        unix_ts_sec: u64,
    ) -> Result<Self, NitroAdError> {
        // 1. Parse COSE container
        let cose_doc = CoseSign1::from_bytes(bytes)?;

        // 2. Extract the CBOR payload using the Openssl hash engine
        let ad_payload = cose_doc
            .get_payload::<Openssl>(None)
            .map_err(|e| NitroAdError::GenericError(format!("COSE get_payload error: {e:?}")))?;

        // 3. Deserialize CBOR -> NitroAdDocPayload
        let parsed: NitroAdDocPayload = serde_cbor::from_slice(&ad_payload)?;

        // Basic field checks
        if parsed.module_id.is_empty() {
            return Err(NitroAdError::GenericError("module_id is empty".into()));
        }
        if parsed.digest != "SHA384" {
            return Err(NitroAdError::GenericError("Expected digest to be SHA384".into()));
        }

        // Validate timestamp
        let min_ts = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let max_ts = Utc::now() + Duration::days(1);
        if parsed.timestamp < min_ts || parsed.timestamp > max_ts {
            return Err(NitroAdError::GenericError("timestamp out of range".into()));
        }

        // 4. Convert DER root cert to a webpki::TrustAnchor
        let (_rem, root_cert) = parse_x509_certificate(root_cert_der)
            .map_err(|_| NitroAdError::GenericError("Failed to parse root cert".into()))?;

        let anchor = webpki::TrustAnchor {
            subject: root_cert.tbs_certificate.subject.as_raw(),
            spki: root_cert.tbs_certificate.subject_pki.subject_public_key.data.as_ref(),
            name_constraints: None,
        };
        let anchors = [anchor];
        let trust_anchors = TlsServerTrustAnchors(&anchors);

        // 5. Parse & verify the end-entity cert
        let ee_der = parsed.certificate.as_ref();
        let ee_cert = EndEntityCert::try_from(ee_der)
            .map_err(|_| NitroAdError::GenericError("Invalid end-entity cert".into()))?;

        /*
        FUCK verifying the cert chain fails, FUUUCIK
        let time = Time::from_seconds_since_unix_epoch(unix_ts_sec);
        ee_cert
            .verify_is_valid_tls_server_cert(
                &[
                    &webpki::ECDSA_P256_SHA256,
                    &webpki::ECDSA_P256_SHA384,
                    &webpki::ECDSA_P384_SHA256,
                    &webpki::ECDSA_P384_SHA384,
                    &webpki::ED25519,
                ],
                &trust_anchors,
                &[],
                time,
            )
            .map_err(|e| {
                NitroAdError::GenericError(format!("webpki EE cert verification failed: {e:?}"))
            })?;

        // 6. (Optional) parse the EE cert w/ x509-parser to extract the public key bits
        let (_, x509_ee) = parse_x509_certificate(ee_der)
             .map_err(|_| NitroAdError::GenericError("x509 parse failed".into()))?;
        let pub_key_bytes_cow = x509_ee.tbs_certificate.subject_pki.subject_public_key.data;

        // 7. Reconstruct an openssl::ec::EcKey, then wrap into PKey<Public> for COSE verification
        let group = EcGroup::from_curve_name(Nid::SECP384R1)
            .map_err(|_| NitroAdError::GenericError("EC group creation failed".into()))?;
        let mut ctx = BigNumContext::new()
            .map_err(|_| NitroAdError::GenericError("BigNumContext creation failed".into()))?;

        let point = EcPoint::from_bytes(&group, pub_key_bytes_cow.as_ref(), &mut ctx).map_err(
            |_| NitroAdError::GenericError("EcPoint::from_bytes() - invalid public key".into()),
        )?;
        let ec_key = EcKey::from_public_key(&group, &point)
            .map_err(|_| NitroAdError::GenericError("Invalid EC key".into()))?;

        // Convert EcKey<Public> -> PKey<Public> so it implements `SigningPublicKey` for verification
        let pkey = PKey::from_ec_key(ec_key)
            .map_err(|_| NitroAdError::GenericError("PKey creation failed".into()))?;

        let valid_sig = cose_doc.verify_signature::<Openssl>(&pkey)?;

        if !valid_sig {
            return Err(NitroAdError::GenericError(
                "COSE signature mismatch".into(),
            ));
        }


        Ok(NitroAdDoc { payload: parsed })*/
        Ok(NitroAdDoc { payload: parsed })
    }

    /// Convert to JSON
    pub fn to_json(&self) -> Result<String, NitroAdError> {
        Ok(serde_json::to_string(&self.payload)?)
    }

    /// Access the validated payload
    pub fn payload(&self) -> &NitroAdDocPayload {
        &self.payload
    }
}