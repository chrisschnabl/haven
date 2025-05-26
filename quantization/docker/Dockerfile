FROM debian:bookworm-slim

ARG BRANCH=master

WORKDIR /workdir

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
     wget git git-lfs ca-certificates build-essential cmake \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 3) Clone/ build llama.cpp/quantize
RUN git clone --branch "$BRANCH" --single-branch \
       https://github.com/ggerganov/llama.cpp.git llama.cpp \
 && cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF \
 && cmake --build llama.cpp/build --target llama-quantize -j$(nproc)

WORKDIR /workdir/llama.cpp/build

# Fetch a smaller <1 GiB model 
RUN wget https://huggingface.co/mradermacher/pythia-70m-GGUF/resolve/main/pythia-70m.f16.gguf
RUN mv pythia-70m.f16.gguf gpt2.f16.gguf

# Debug model/binary exist 
#RUN echo "=== build dir contents ===" \
# && ls -lah . \
# && echo "=== bin contents ===" \
# && ls -lah bin \
# && echo "=== ldd llama-quantize ===" \
# && ldd bin/llama-quantize || true

ENV LD_LIBRARY_PATH=/workdir/llama.cpp/build/lib

ENTRYPOINT [ "bash", "-c", "\
    echo '=== app contents ==='; \
    ls -lah .; \
    if [ -f gpt2.f16.gguf ]; then \
      echo 'Model present, running quantize…'; \
      exec bin/llama-quantize gpt2.f16.gguf gpt2.Q4.gguf Q4_K_M; \
    else \
      echo 'ERROR: gpt2.f16.gguf not found'; \
    fi; \
    echo 'Entering wait loop…'; \
    timeout 20s bash -c 'while true; do sleep 3600; done' \
"]
