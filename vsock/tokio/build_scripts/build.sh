#!/bin/bash
ARCH=$(uname -m)
RUST_DIR=$(dirname "$(readlink -f "$0")")

cargo build --manifest-path="${RUST_DIR}/Cargo.toml" --target="${ARCH}-unknown-linux-musl" --release
cp "${RUST_DIR}/target/${ARCH}-unknown-linux-musl/release/vsock-sample" "${RUST_DIR}"