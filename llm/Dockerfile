# syntax=docker/dockerfile:1
FROM ubuntu:22.04@sha256:1cc7bb38a74c0e126716646e47c0b3c5c139547d386d5ff7a64cf3ea316ca523 AS toolchain
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

RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
    libcublas-12-6=12.6.4.1-1 libcublas-dev-12-6=12.6.4.1-1 \
    && rm -rf /var/lib/apt/lists/*

FROM toolchain AS source
ARG LLAMA_COMMIT=b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
ARG LLAMA_SOURCE_SHA256=03fb04316eb32a7b7347004a79a8dd60531c02f5a064d853fed3b53828951723
RUN curl -fSL --retry 5 "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${LLAMA_COMMIT}" -o /tmp/llama.tar.gz \
    && echo "${LLAMA_SOURCE_SHA256}  /tmp/llama.tar.gz" | sha256sum -c - \
    && mkdir /llama && tar -xzf /tmp/llama.tar.gz -C /llama --strip-components=1 \
    && rm /tmp/llama.tar.gz

# Build the matching upstream UI from its npm lockfile; no Node in the runtime.
FROM node:24-bookworm-slim@sha256:24b8bc17702002d2ed0c1da9ad66c3ee507cc279d0856726662ff2b6fc35c149 AS ui
WORKDIR /ui
COPY --from=source /llama/tools/ui/ ./
RUN npm ci --no-audit --no-fund
RUN LLAMA_UI_OUT_DIR=/ui/dist LLAMA_BUILD_NUMBER=v0.4.1 npm run build

FROM source AS build
ARG BUILD_JOBS=6
RUN cmake -S /llama -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=OFF \
      -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod \
      -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FA=OFF \
      -DGGML_CUDA_NCCL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_BUILD_IS_DEV=OFF \
    && cmake --build /build --target ggml-cuda llama --parallel "${BUILD_JOBS}"
COPY --from=ui /ui/dist/ /llama/tools/ui/dist/
RUN cmake --build /build --target llama-server --parallel "${BUILD_JOBS}" \
    && mkdir -p /out/bin /out/lib /out/licenses /out/info \
    && cp -a /build/bin/llama-server /build/bin/*.so* /out/bin/ \
    && cp -a /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so* /out/lib/ \
    && cp -a /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublasLt.so* /out/lib/ \
    && cp /llama/LICENSE /out/licenses/llama.cpp-LICENSE \
    && cp /usr/share/doc/libcublas-12-6/copyright /out/licenses/cuBLAS-LICENSE \
    && cp /llama/tools/ui/package-lock.json /out/info/ui-package-lock.json \
    && dpkg-query -W > /out/info/build-packages.txt \
    && printf 'llama.cpp: v0.4.1\nCommit: %s\nCUDA: 12.6.3\ncuBLAS: 12.6.4.1\nGPU: sm_87 + compute_87\n' "$LLAMA_COMMIT" > /out/info/build-info.txt

FROM source AS model
ARG MODEL_REVISION=daeb8e2d528a760970442092f6bf1e55c3b659eb
ARG MODEL_SHA256=d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5
RUN mkdir /models \
    && curl -fSL --retry 5 "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/${MODEL_REVISION}/Qwen3-1.7B-Q4_K_M.gguf?download=true" -o /models/Qwen3-1.7B-Q4_K_M.gguf \
    && echo "${MODEL_SHA256}  /models/Qwen3-1.7B-Q4_K_M.gguf" | sha256sum -c - \
    && curl -fsSL --retry 5 https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/90862c4b9d2787eaed51d12237eafdfe7c5f6077/LICENSE -o /models/LICENSE \
    && printf 'Qwen3-1.7B Q4_K_M\nRepository: ggml-org/Qwen3-1.7B-GGUF\nRevision: %s\nSHA256: %s\n' "$MODEL_REVISION" "$MODEL_SHA256" > /models/model-info.txt

FROM alexturner/workload-cuda-debug@sha256:501228b28f1792a84ab978e1e138be0283f04aa21e839d67f2ed001924010348
LABEL org.opencontainers.image.title="local-llm" \
      org.opencontainers.image.description="Qwen3 1.7B CUDA chat on Admiral Jetson Orin, Ubuntu 26.04" \
      org.opencontainers.image.version="2026-09-20-qwen3-1.7b-r1" \
      co.admrl.workload="local-llm"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/ /opt/llama/bin/
COPY --from=build /out/lib/ /usr/local/lib/
COPY --from=build /out/licenses/ /usr/share/licenses/local-llm/
COPY --from=build /out/info/ /opt/llama/info/
COPY --from=model /models/ /opt/llama/models/
COPY llm/start-local-llm llm/serve-local-llm /usr/local/bin/
COPY llm/README.md /opt/llama/README.md
RUN chmod 0755 /usr/local/bin/start-local-llm /usr/local/bin/serve-local-llm \
    && printf '/opt/llama/bin\n' > /etc/ld.so.conf.d/llama.conf \
    && ldconfig \
    && test -z "$(find /usr -name 'libcuda.so*' -print -quit)" \
    && test -z "$(find /usr -type d -name stubs -print -quit)" \
    && printf '\nLocal LLM: http://DEVICE_IP:8080\nLogs: /var/log/local-llm/server.log\nCUDA diagnostics: cuda-debug smoke\n' >> /etc/motd
ENV CUDA_SMOKE_ON_BOOT=0 LLM_HOST=0.0.0.0 LLM_PORT=8080 LLM_CONTEXT=2048 LLM_THREADS=4
WORKDIR /opt/llama
EXPOSE 8080/tcp 22/tcp
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/start-local-llm"]
