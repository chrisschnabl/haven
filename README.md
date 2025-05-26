# ⚓ Haven

The following repository contains all artifacts for this MPhil project, which is a mix of rust and python (and too many scripts too glue stuff together). A good starting point is to look at root `Makefile` and the `README.md` files in all the subdirectories:

- `enclave` - the main part of the project, the rust code for running the enclave and the host, sending files, running llama in the enclave, generating attestation documents
- `verify_attestation` - the python code for verifying the attestation
- `toxicity` - the python code for running a toxicity classifier
- `llama_runner` - the rust code for running llama using llama-cpp2
- `quantization` - quantize models in AWS Nitro Enclaves (through llama.cpp and pytorch)
- `scripts` - the ducttape basically
- `vsock` - 