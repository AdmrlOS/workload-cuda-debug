# syntax=docker/dockerfile:1
ARG BUILD_IMAGE=ubuntu:22.04@sha256:1cc7bb38a74c0e126716646e47c0b3c5c139547d386d5ff7a64cf3ea316ca523
ARG UBUNTU_IMAGE=ubuntu:26.04@sha256:e03767b4dc7cb87fc57b1f119d40d9a997ecc5570dc6e22d57d6b7e333bbe78c
# Keep CUDA 12.6's compiler environment fixed while testing newer userspace.
FROM ${BUILD_IMAGE} AS cuda-build
ARG TARGETARCH
RUN test "$TARGETARCH" = arm64
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl g++ \
    && rm -rf /var/lib/apt/lists/*
# NVIDIA's Jetson ARM64 repository, not the SBSA/desktop driver repository.
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && echo '2778cdec0d834b2f895ccef906ff1d33a49acd9a3fe3bf798b67475b9338eff7  /tmp/cuda-keyring.deb' | sha256sum -c - \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
       cuda-nvcc-12-6=12.6.85-1 cuda-nvvm-12-6=12.6.85-1 cuda-crt-12-6=12.6.85-1 \
       cuda-cudart-12-6=12.6.77-1 cuda-cudart-dev-12-6=12.6.77-1 \
       cuda-driver-dev-12-6=12.6.77-1 cuda-cccl-12-6=12.6.77-1 \
    && rm -rf /var/lib/apt/lists/*
COPY src/ /src/
RUN mkdir -p /out/lib /out/licenses \
    && g++ -std=c++17 -O2 -g -Wall -Wextra -Werror -I/usr/local/cuda-12.6/include \
       /src/driver-probe.cpp -ldl -o /out/cuda-driver-probe \
    && /usr/local/cuda-12.6/bin/nvcc -O2 -lineinfo --cudart shared \
       -gencode arch=compute_87,code=sm_87 -gencode arch=compute_87,code=compute_87 \
       /src/vector-add.cu -o /out/cuda-vector-add \
    && cp -a /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so* /out/lib/ \
    && cp /usr/share/doc/cuda-cudart-12-6/copyright /out/licenses/CUDA-LICENSE \
    && dpkg-query -W > /out/build-packages.txt \
    && /usr/local/cuda-12.6/bin/nvcc --version > /out/nvcc-version.txt

FROM ${UBUNTU_IMAGE}
ARG IMAGE_VERSION=2026-09-20-ubuntu26.04-cuda12.6.3-r1
LABEL org.opencontainers.image.title="cuda-debug" \
      org.opencontainers.image.description="Admiral Jetson Orin CUDA diagnostics and SSH on port 22" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      co.admrl.workload="cuda-debug"
ENV LANG=C.UTF-8 CUDA_SMOKE_ON_BOOT=1 CUDA_SMOKE_TIMEOUT=60
# The compiler, driver stubs, and cuda-compat packages stay out of this stage.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       openssh-server tini ca-certificates curl jq less nano procps iproute2 iputils-ping \
       strace gdb binutils file util-linux libstdc++6 \
    && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* \
    && groupadd --gid 28 admiral-video \
    && useradd --create-home --shell /bin/bash --groups admiral-video cuda \
    && mkdir -p /root/.ssh /home/cuda/.ssh /opt/cuda-debug /var/log/cuda-debug /run/sshd \
    && chmod 0700 /root/.ssh /home/cuda/.ssh
COPY --from=cuda-build /out/lib/ /usr/local/lib/
COPY --from=cuda-build /out/cuda-driver-probe /out/cuda-vector-add /usr/local/bin/
COPY --from=cuda-build /out/licenses/ /usr/share/licenses/cuda-debug/
COPY --from=cuda-build /out/build-packages.txt /out/nvcc-version.txt /opt/cuda-debug/
COPY src/ /opt/cuda-debug/src/
COPY scripts/ /usr/local/bin/
COPY ssh/sshd_config /etc/ssh/sshd_config
COPY ssh/authorized_keys /root/.ssh/authorized_keys
COPY ssh/authorized_keys /home/cuda/.ssh/authorized_keys
COPY README.md /opt/cuda-debug/README.md
RUN chmod 0755 /usr/local/bin/cuda-debug /usr/local/bin/start-cuda-debug \
    && chmod 0600 /root/.ssh/authorized_keys /home/cuda/.ssh/authorized_keys \
    && chown -R cuda:cuda /home/cuda/.ssh \
    && ssh-keygen -lf /root/.ssh/authorized_keys \
    && ldconfig \
    && dpkg-query -W > /opt/cuda-debug/runtime-packages.txt \
    && printf 'CUDA toolkit: 12.6.3\nNVCC: 12.6.85\nlibcudart: 12.6.77 (Jetson ARM64)\nGPU targets: sm_87 + compute_87 PTX\nAdmiral R39.2.1 acceptance: pending hardware test\n' > /opt/cuda-debug/build-info.txt \
    && . /etc/os-release \
    && printf 'Runtime OS: %s\nImage version: %s\nCompiler build OS: Ubuntu 22.04\n' "$PRETTY_NAME" "$IMAGE_VERSION" >> /opt/cuda-debug/build-info.txt \
    && getconf GNU_LIBC_VERSION >> /opt/cuda-debug/build-info.txt \
    && printf '\nCUDA debug: cuda-debug inspect | cuda-debug smoke | cuda-debug jit\nBoot results: /var/log/cuda-debug/boot.log\n' > /etc/motd \
    && test -z "$(find /usr -name 'libcuda.so*' -print -quit)" \
    && test -z "$(find /usr -type d -name stubs -print -quit)"
WORKDIR /opt/cuda-debug
EXPOSE 22/tcp
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/start-cuda-debug"]
