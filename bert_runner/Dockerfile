FROM ubuntu:latest
RUN apt-get update && apt-get install -y libgomp1 wget unzip

# Download version 2.4.0 of libtorch
RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip -d /opt && \
    rm libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip

ENV LIBTORCH=/opt/libtorch
RUN echo "${LIBTORCH}/lib" > /etc/ld.so.conf.d/libtorch.conf && ldconfig
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

COPY ./target/debug/bert_runner_example bert_runner
RUN mkdir model
COPY ./bert_runner/model/ /model/
CMD ["./bert_runner"]
# 20156 bytes in memorym in