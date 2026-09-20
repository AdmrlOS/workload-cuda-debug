# Local LLM on Admiral / Jetson Orin

Ubuntu 26.04.1, CUDA 12.6, llama.cpp v0.4.1 and Qwen3 1.7B Q4_K_M.
The 1,282,439,264-byte model is included in the image; no download or Hugging Face
account is needed on the device. Inference runs locally. This image is larger than
the diagnostic image because it includes the model and cuBLAS.

## Deploy and chat

Deploy `alexturner/workload-local-llm:2026-09-20-qwen3-1.7b-r1` through Admiral as
`local-llm` with `IsolateDevices=false`. Keep its default entrypoint. SSH uses the
same alexanderturner public key on port 22. Allow port 8080 on your trusted LAN.
Open **http://DEVICE_IP:8080** for the bundled llama.cpp chat interface.

Startup first runs the CUDA kernel smoke test as `cuda` (UID 1001, supplementary
GID 28), verifies llama.cpp lists `CUDA0`, then requests all model layers on that
GPU. It refuses to launch the chat service if preflight fails. SSH stays available.

```sh
ssh root@DEVICE_IP
tail -f /var/log/local-llm/server.log
curl http://127.0.0.1:8080/health
```

Loading can return HTTP 503 before `/health` reports `{"status":"ok"}`. Check the
log for CUDA0/Orin, a CUDA model buffer and `offloaded .../... layers to GPU`.
Model output alone is not proof of GPU offload. Generation logs also show timing
and tokens per second; measure on this board instead of assuming performance.

Try: “Write a short pirate poem about a tiny GPU living on the edge.”

An OpenAI-compatible chat endpoint is also available:

```sh
curl http://DEVICE_IP:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Explain CUDA in three sentences."}],"max_tokens":128,"stream":false}'
```

The LAN demo has no API key by default. For access through SSH only, set
`LLM_HOST=127.0.0.1` in Admiral and use
`ssh -L 8080:127.0.0.1:8080 root@DEVICE_IP`, then open http://localhost:8080.
Alternatively configure `LLAMA_API_KEY` in Admiral and enter it in the chat UI;
API requests then require an `Authorization: Bearer ...` header.

## Defaults and memory

- Context: 2,048 tokens shared by input and output; one chat slot.
- Four CPU threads; 256-token batch, 128-token microbatch.
- Qwen thinking mode disabled for short interactive answers.
- Model layers requested on CUDA0; automatic memory fitting disabled.
- Flash attention and CUDA graphs disabled for this first integration.
- CUDA VMM disabled at build time; use the allocation path already smoke-tested.

The Orin's CPU and GPU share RAM. The model file size is not the total working-set
size: context, compute buffers, driver allocations and file cache also consume
memory. Observe `free -h` and server allocation logs. `LLM_CONTEXT`, `LLM_THREADS`,
`LLM_HOST` and `LLM_PORT` may be set in Admiral. Start with the defaults.
No agent tools, shell execution or application actions are enabled in the server.

## Rebuild

```sh
docker buildx build --platform linux/arm64 --load -f llm/Dockerfile \
  -t alexturner/workload-local-llm:2026-09-20-qwen3-1.7b-r1 \
  -t alexturner/workload-local-llm:latest .
docker push alexturner/workload-local-llm:2026-09-20-qwen3-1.7b-r1
docker push alexturner/workload-local-llm:latest
```

`BUILD_JOBS` defaults to six; reduce it on memory-limited build hosts. Compilation
uses the pinned Ubuntu 22.04 / CUDA 12.6 Jetson toolchain and SM 8.7. The runtime
inherits the Ubuntu 26.04 diagnostic image whose four GPU tests passed in the
user's supplied logs. It includes no NVIDIA driver or runtime CUDA stubs.
The chat UI is built from the same source commit using its npm lockfile.

Pinned inputs:

- llama.cpp commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (v0.4.1).
- Model repository `ggml-org/Qwen3-1.7B-GGUF`, revision
  `daeb8e2d528a760970442092f6bf1e55c3b659eb`.
- Model SHA-256 `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`.
- cuBLAS `12.6.4.1-1`; libcudart inherited at `12.6.77`.

Licenses: llama.cpp MIT, Qwen Apache-2.0, CUDA/cuBLAS NVIDIA license. Copies are
under `/usr/share/licenses/local-llm` and `/opt/llama/models/LICENSE`.

Sources: [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF),
[quantized model](https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF).

The user reports this LLM container running successfully on the Jetson. Detailed
GPU offload and performance logs remain to be recorded; see [VALIDATION.md](VALIDATION.md).
Local CPU inference checks validate packaging/API only.
