FROM ubuntu:latest
RUN apt-get update && apt-get install -y libgomp1 wget unzip

COPY ./target/debug/haven-enclave haven-enclave

# TODO CS: potentially archive such critical dependencies, i.e. what if libtorch removes this
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip -d /opt && \
    rm libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip

ENV LIBTORCH=/opt/libtorch
RUN echo "${LIBTORCH}/lib" > /etc/ld.so.conf.d/libtorch.conf && ldconfig
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

CMD ["./haven-enclave"]