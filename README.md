# ⚓ Haven

The following repository contains all artifacts for this MPhil project, which is a mix of rust and python (and too many scripts too glue stuff together). A good starting point is to look at root `Makefile` and the `README.md` files in all the subdirectories:

- `analysis` - code and artefacts to reproduce plots in my thesis based on benchmark runs
- `enclave` - first prototype for running the enclave and the host, sending files, running llama in the enclave, generating attestation documents. Does not include AI Safety benchmarks.
- `llama_runner` - running llama et al. using llama-cpp2 through the actor model
- `bert_runner` - running bert models using rust-bert through the actor model (compiles against mock implementation by default or links with libpytdorch if compiled through `-F use_rust_bert`)
- `evaluation` - scaffolding (messages, file_transfer, dataset handling) to run AI safety benchmarks in haven. Configurations and prompts for the audit code for different graphs is included in `evaluation/tasks`. Recreates huggingfaces dataset library in dataset.rs. Includes code to run the evaluation without
- `evaluation_encalve` - protocol for the enclave side. uses type-state pattern: `InitializedState -> LlamaLoadedState -> BertLoadedState -> DatasetLoadedState -> EvaluatedState -> AttestedState`
- `evaluation_host` - protocol for the host side. uses type-state-pattern: `Disconnected -> Connected -> LlamaSent -> BertSent -> DatasetSent -> EvaluationComplete -> AttestationReceived`
- `quantization` - quantize models in AWS Nitro Enclaves (through llama.cpp and pytorch)
- `scripts` - the ducttape (basically)
- `vsock` - abstracts away (most) of the vsock trouble, takes a server/client handle that interact with the sock
- `gpu_baseline` - esentially `evaluation` but in Python to run on GPUs through vLLM