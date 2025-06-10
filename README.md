# ⚓ Haven

A secure AI evaluation framework for running AI safety benchmarks within AWS Nitro Enclaves, combining Rust and Python components.

## Overview

The following repository contains all artifacts for this MPhil project, which is a mix of rust and python (and too many scripts to glue stuff together). A good starting point is to look at root `Makefile` and the `README.md` files in all the subdirectories:

- `analysis` - code and artefacts to reproduce plots in my thesis based on benchmark runs
- `enclave` - first prototype for running the enclave and the host, sending files, running llama in the enclave, generating attestation documents. Does not include AI Safety benchmarks.
- `llama_runner` - running llama et al. using llama-cpp2 through the actor model
- `bert_runner` - running bert models using rust-bert through the actor model (compiles against mock implementation by default or links with libpytdorch if compiled through `-F use_rust_bert`)
- `evaluation` - scaffolding (messages, file_transfer, dataset handling) to run AI safety benchmarks in haven. Configurations and prompts for the audit code for different graphs is included in `evaluation/tasks`. Recreates huggingfaces dataset library in dataset.rs. Includes code to run the evaluation without
- `evaluation_enclave` - protocol for the enclave side. uses type-state pattern: `InitializedState -> LlamaLoadedState -> BertLoadedState -> DatasetLoadedState -> EvaluatedState -> AttestedState`
- `evaluation_host` - protocol for the host side. uses type-state-pattern: `Disconnected -> Connected -> LlamaSent -> BertSent -> DatasetSent -> EvaluationComplete -> AttestationReceived`
- `quantization` - quantize models in AWS Nitro Enclaves (through llama.cpp and pytorch)
- `scripts` - the ducttape (basically)
- `vsock` - abstracts away (most) of the vsock trouble, takes a server/client handle that interact with the sock
- `gpu_baseline` - essentially `evaluation` but in Python to run on GPUs through vLLM

## Run Enclave

Make sure you have AWS Nitro CLI and SDK installed [here](https://aws.amazon.com/ec2/nitro/) and sufficient memory allocated in the nitro allocator.

From the project directory:
```bash
cd enclave && cargo build 
make build-docker && make build-eif && make run-enclave
```

Run `make terminate-enclave` if you wish to stop the enclave.

## Run Host

```bash
cd enclave && cargo build && cargo run
```

## Analysis

Make sure to have [uv](https://github.com/astral-sh/uv) installed.

All the raw data is in `quantization_ablation_model` for local analysis and in the `remote_experiments` for AWS. Make sure input datasets contain classification pairs for postprocessing.

```bash
cd analysis
uv run local_analysis.py
uv run aws_analysis.py
```

### AWS Analysis


- [Enclave Combined Analysis](analysis/aws-analysis-2025-06-09-17-05-33/enclave_combined_analysis.pdf)
- [Token Distribution](analysis/aws-analysis-2025-06-09-17-05-33/token_distribution.pdf)

### Local Analysis


- [Classification Accuracy](analysis/local-analysis-2025-06-09-16-52-38/classification_accuracy_valid_only.pdf)
