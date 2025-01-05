# Copyright 2019 fsyncd, Berlin, Germany.
# Additional material, copyright of the containerd authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARCH := $(shell uname -m)
TOOLCHAIN ?= $(ARCH)-unknown-linux-musl
RUST_DIR := $(shell readlink -m $(shell dirname $(firstword $(MAKEFILE_LIST))))
VPATH = target/system target/$(TOOLCHAIN)/debug target/$(TOOLCHAIN)/release

SRCS := $(shell find . -type f -name '*.rs' | grep -v 'tests')
ID := $(shell date +%s)
SERVER_BIN = test_server
CLIENT_BIN = test_client

.PHONY: all check clean tokio_vsock kmod vm build

all: fmt clippy tokio_vsock test_server

check: tokio_vsock test

test:
	cargo test --target=$(TOOLCHAIN) --all

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

clean:
	rm -rf $(RUST_DIR)/target $(RUST_DIR)/vsock_sample_*.eif $(RUST_DIR)/vsock-sample

build:
	curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
	rustup target install $(TOOLCHAIN)
	cargo build --manifest-path=$(RUST_DIR)/Cargo.toml --target=$(TOOLCHAIN) --release

tokio_vsock: $(SRCS)
	cargo build --lib

tokio_vsock_all_features: $(SRCS)
	cargo build --lib --all-features

tokio_vsock_min_deps: $(SRCS)
	cargo build --lib --all-features -Z direct-minimal-versions

test_server: tokio_vsock test_server/src/main.rs
	cargo build --manifest-path=test_server/Cargo.toml --target=$(TOOLCHAIN) --release

test_client: tokio_vsock test_client/src/main.rs
	cargo build --manifest-path=test_client/Cargo.toml --target=$(TOOLCHAIN) --release

run-server:
	./test_server/target/aarch64-unknown-linux-musl/release/test_server --listen 1337

# Run the client
run-client:
	./test_client/target/aarch64-unknown-linux-musl/release/test_client --port 1337

# Help target to list available commands
help:
	@echo "Available commands:"
	@echo "  make build          - Build the server and client binaries"
	@echo "  make run-server     - Run the Virtio socket test server"
	@echo "  make run-client     - Run the Virtio socket test client"
	@echo "  make clean          - Clean up the project"
	@echo "  make help           - Show this help message"