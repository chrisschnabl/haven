FROM ubuntu:latest
RUN apt-get update && apt-get install -y libgomp1 wget unzip

# Download version 2.4.0 of libtorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip -d /opt && \
    rm libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip

ENV LIBTORCH=/opt/libtorch
RUN echo "${LIBTORCH}/lib" > /etc/ld.so.conf.d/libtorch.conf && ldconfig
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

COPY ./target/release/bert_runner_example bert_runner

CMD ["./bert_runner"]