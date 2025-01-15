# Enclave

## Build 

### Prerequisites

- Rust 
- EC2 with Nitro Enclaves enabled and nitro-cli installed, see `scripts/ec2` for more context.
- Download a (quantized) llama model of your choice, e.g. `wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf?download=true`

```bash
make build && build-docker && make build-eif
```

## Run 

### Host

```bash
./target/release/enclave --mode host --port 5005 --cid 16 --file model.gguf
./target/release/enclave --mode host --port 5005 --cid 16 --prompt "Hello, how are you?"
```

### Enclave

```bash
make run-enclaves
```
