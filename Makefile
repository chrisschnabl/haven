BIN_NAME = haven

# TODO CS: include scripts into makefile

# 'all' target: run formatting, linting, and build
.PHONY: all
all: build build-docker build-eif #fmt clippy build 

# 'check' target: build + test
.PHONY: check
check: build test

# Format code
.PHONY: fmt
fmt:
	cargo fmt --all -- --check

# Run clippy with all targets, all features, and deny warnings
.PHONY: clippy
clippy:
	cargo clippy --all-targets --all-features -- -D warnings

# Build in release mode
.PHONY: build
build:
	cargo build --workspace --release && cp target/release/enclave haven

# Test everything in the workspace
.PHONY: test
test:
	cargo test --workspace

# Clean the entire target directory
.PHONY: clean
clean:
	cargo clean

# Build the Docker image for your enclave
.PHONY: build-docker
build-docker:
	docker build -f enclave/Dockerfile.enclave -t enclave .

# Build an EIF
.PHONY: build-eif
build-eif:
	nitro-cli build-enclave --docker-uri enclave:latest --output-file enclave.eif

# Run an enclave
.PHONY: run-enclave
run-enclave:
	@echo "Starting the enclave..."
	@ENCLAVE_ID=$$(nitro-cli run-enclave \
			--cpu-count 2 \
			--memory 10000 \
			--enclave-cid 16 \
			--eif-path enclave.eif \
			--debug-mode | jq -r '.EnclaveID') && \
	if [ -n "$$ENCLAVE_ID" ]; then \
		echo "Enclave started with ID: $$ENCLAVE_ID"; \
		echo "Connecting to the enclave console..."; \
		nitro-cli console --enclave-id $$ENCLAVE_ID; \
	else \
		echo "Failed to retrieve EnclaveID"; \
		exit 1; \
	fi

.PHONY: terminate-enclaves
terminate-enclaves:
	nitro-cli describe-enclaves \
	| jq -r '.[].EnclaveID' \
	| xargs -I {} nitro-cli terminate-enclave --enclave-id {}

# Restart the Nitro Enclaves Allocator service
.PHONY: restart-alloc
restart-alloc:
	sudo systemctl restart nitro-enclaves-allocator.service

# Run your server-side code (inside the enclave)
.PHONY: run-server
run-server:
	./target/release/$(BIN_NAME) --mode enclave --port 1337

# Run your client-side code (outside the enclave)
.PHONY: run-client
run-client:
	./target/release/$(BIN_NAME) --mode host --port 1337 --file example.txt

# Help text
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make build          - Build all crates in release mode (workspace)"
	@echo "  make run-server     - Run the server code in enclave mode"
	@echo "  make run-client     - Run the client code in host mode"
	@echo "  make test           - Run all tests"
	@echo "  make clean          - Clean up the project"
	@echo "  make fmt            - Format the code"
	@echo "  make clippy         - Lint the code"
	@echo "  make build-docker   - Build the Docker image for Nitro Enclave"
	@echo "  make build-eif      - Build the EIF for Nitro Enclave"
	@echo "  make run-enclave    - Run the enclave (launch and attach console)"
	@echo "  make terminate-enclaves -- Terminate all running enclaves"
	@echo "  make help           - Show this help message"