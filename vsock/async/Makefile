ARCH := $(shell uname -m)
TOOLCHAIN ?= $(ARCH)-unknown-linux-musl
RUST_DIR := $(shell readlink -m $(shell dirname $(firstword $(MAKEFILE_LIST))))
VPATH = target/$(TOOLCHAIN)/debug target/$(TOOLCHAIN)/release

SRCS := $(shell find . -type f -name '*.rs' | grep -v 'tests')
BIN_NAME = test_program

.PHONY: all check clean build run-server run-client help

all: fmt clippy build

check: build test

test:
	cargo test --target=$(TOOLCHAIN) --all

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

clean:
	rm -rf $(RUST_DIR)/target

install:
	curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
	rustup target install $(TOOLCHAIN)

build:
	cargo build --manifest-path=$(RUST_DIR)/Cargo.toml --target=$(TOOLCHAIN) --release

build-enclave:
	../../scripts/ec2/

run-server:
	./target/$(TOOLCHAIN)/release/$(BIN_NAME) --mode enclave --port 1337


run-client:
	./target/$(TOOLCHAIN)/release/$(BIN_NAME) --mode host --port 1337 --file example.txt

help:
	@echo "Available commands:"
	@echo "  make build          - Build the test program binary"
	@echo "  make run-server     - Run the test program in server mode"
	@echo "  make run-client     - Run the test program in client mode"
	@echo "  make clean          - Clean up the project"
	@echo "  make fmt            - Format the code"
	@echo "  make clippy         - Lint the code"
	@echo "  make help           - Show this help message"