import base64
import cbor2
from cose.messages import CoseMessage
from cryptography import x509
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec

def main():
    print("Hello from attestation!")

    # Load the attestation document (assumed to be base64 encoded)
    attestation_doc_b64 = "..."  # Replace with actual base64 attestation document
    attestation_doc = base64.b64decode(attestation_doc_b64)

    # Decode the CBOR-encoded attestation document
    cose_msg = CoseMessage.decode(attestation_doc)

    # Extract the payload (attestation document)
    attestation_payload = cbor2.loads(cose_msg.payload)

    # Extract the certificate
    cert_der = attestation_payload['certificate']
    cert = x509.load_der_x509_certificate(cert_der)

    # Verify the certificate against the AWS Nitro Enclaves Root Certificate
    # Load the AWS Nitro Enclaves Root Certificate
    with open("AWS_NitroEnclaves_Root-G1.pem", "rb") as f:
        root_cert = x509.load_pem_x509_certificate(f.read())

    # Create a certificate store and add the root certificate
    store = x509.CertificateStore()
    store.add_certificate(root_cert)

    # Create a certificate store context for verification
    store_ctx = x509.CertificateStoreContext(store, cert)
    store_ctx.verify_certificate()

    # Verify the COSE message signature using the public key from the certificate
    public_key = cert.public_key()
    cose_msg.key = public_key
    cose_msg.verify_signature()

    # Extract and verify user_data
    user_data = attestation_payload.get('user_data')
    if user_data:
        # Assuming user_data contains concatenated SHA-384 hashes of input, output, and model_id
        expected_input_hash = ...
        expected_output_hash = ...
        expected_model_id_hash = ...

        input_hash = user_data[:48]
        output_hash = user_data[48:96]
        model_id_hash = user_data[96:144]

        assert input_hash == expected_input_hash, "Input hash mismatch"
        assert output_hash == expected_output_hash, "Output hash mismatch"
        assert model_id_hash == expected_model_id_hash, "Model ID hash mismatch"
    else:
        raise ValueError("user_data not found in attestation document")


if __name__ == "__main__":
    main()
