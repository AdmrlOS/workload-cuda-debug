# Local LLM validation — 2026-09-20

Build: `alexturner/workload-local-llm:2026-09-20-qwen3-1.7b-r1`.
Published as the dated tag above and `latest`. OCI manifest digest:
`sha256:789640cb026870b7cd0d71a78a1db0e951a4f7c8e9786121b1e95c75e4e1e38f`.
Docker reports 1,505,652,199 bytes for this ARM64 image.

Immutable deployment reference:

```text
alexturner/workload-local-llm@sha256:789640cb026870b7cd0d71a78a1db0e951a4f7c8e9786121b1e95c75e4e1e38f
```

`bash llm/test-local.sh` passed on ARM64 Docker Desktop:

- Model SHA-256 matches the pinned Hugging Face LFS identity.
- Server and CUDA backend libraries resolve inside the final Ubuntu 26.04 image.
- No bundled NVIDIA driver, development stubs, or private SSH host keys.
- Normal startup refuses inference without a GPU; root and non-root SSH still work.
- The non-root account retains supplementary numeric GID 28.
- A separate, explicitly CPU-only test container loads the model, serves the
  embedded chat HTML, and returns generated tokens through `/v1/chat/completions`.
- The test prompt was “Say hello in one short sentence.” The response was
  “Hello! How can I assist you today?”
- SSH-only startup and the standalone CPU test server stop cleanly.

No CPU override is enabled by the production entrypoint. The CPU run proves
model/API packaging only. Browser interaction was not manually tested.

The prior Ubuntu 26.04 diagnostic image passed four CUDA tests in user-provided
Jetson logs. The user subsequently reported that this LLM container is running
successfully on the Jetson. This is user-reported application validation; detailed
LLM layer-offload logs, memory measurements, generation speed and restart results
have not yet been captured in this repository.

After deployment, open port 8080 on the trusted LAN, wait for `/health`, and
record `/var/log/local-llm/server.log`. Confirm CUDA0/Orin, CUDA model allocations,
and the reported number of offloaded layers before treating an answer as a GPU
inference pass. Save prompt/generation timing from a real chat request.
