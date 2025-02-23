FROM ubuntu:latest
RUN apt-get update && apt-get install -y libgomp1 wget unzip

# Download and unzip the CPU version of libtorch v2.4 (adjust URL if needed)

RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip -d /opt && \
    rm libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip

# Set the LIBTORCH environment variable
ENV LIBTORCH=/opt/libtorch

# (Optional) Debug: list the libtorch_cpu libraries
RUN find ${LIBTORCH} -name "libtorch_cpu.so*"

# Create a symlink for libtorch_cpu.so based on the actual versioned file
#RUN cd ${LIBTORCH}/lib && \
#    ln -sf $(ls libtorch_cpu.so.* | head -n 1) libtorch_cpu.so

# Register libtorch libraries with the linker and update ldconfig
RUN echo "${LIBTORCH}/lib" > /etc/ld.so.conf.d/libtorch.conf && ldconfig

# Update the library path
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Copy your binary and set the entrypoint
COPY ./target/debug/bert_runner_example bert_runner
RUN mkdir model
COPY ./bert_runner/model/ /model/
CMD ["./bert_runner"]
# 20156