#!/bin/bash
ARCH=$(uname -m)
RUST_DIR=$(dirname "$(readlink -f "$0")")

# Install Rust
if ! command -v rustup &> /dev/null; then
    echo "Installing rustup..."
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "Rustup is already installed."
fi


rustup target add "${ARCH}-unknown-linux-musl"