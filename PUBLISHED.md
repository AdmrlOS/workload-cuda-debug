# Published image — 2026-09-20

Repository: https://hub.docker.com/r/alexturner/workload-cuda-debug

## Ubuntu 22.04 baseline

Tags: `2026-09-20-cuda12.6.3-r2` and `latest`.

Revision 2 changes SSH to port **22**. The original dated tag remains on port
2222, which collided with an existing listener on the deployed device.

Immutable deployment reference:

```text
alexturner/workload-cuda-debug@sha256:fb9eb53b01dd1da62e96ee34613b7e174fa60c5feeab079d656e156a9d2a7e37
```

Platform: `linux/arm64`. Vanilla Ubuntu 22.04 base, CUDA 12.6.3 compiler,
libcudart 12.6.77, Orin `sm_87` machine code plus `compute_87` PTX.

Local verification: `bash test-local.sh` passed on ARM64 Docker Desktop.
Verified shared runtime dependencies, absence of bundled NVIDIA driver/stubs and
SSH host keys, nonzero boot smoke result without a driver, successful SSH public
key login on port 22 as root and cuda, numeric supplementary GID 28, and graceful shutdown.
SSH login tests used an ephemeral key added only to a disposable test container;
the published image contains only the pinned alexanderturner GitHub public key.

User-supplied Admiral/Jetson logs on 2026-09-20 confirm root and non-root smoke
tests and forced PTX JIT with the cache disabled. The driver was loaded from
`/run/admiral/nvidia/lib/libcuda.so.1`, reported API 13.2, and enumerated Orin SM
8.7. Each execution verified all 1,048,579 vector-add results. The non-root process
reported UID/GID 1000 and supplementary GID 28. The smoke boot exit status was 0.
These are reported hardware results, not tests run remotely by this build session.
The supplied logs did not independently display the deployed image digest.

Recreation, reboot/cold-boot repetition and kernel-log review remain unverified.
Deploy through Admiral as `cuda-debug` with `IsolateDevices=false`, then follow
the acceptance checks in README.md. `latest` remains on this Ubuntu 22.04 image.

## Ubuntu 26.04 proof of concept

Tags: `2026-09-20-ubuntu26.04-cuda12.6.3-r1` and `ubuntu26.04`.

```text
alexturner/workload-cuda-debug@sha256:501228b28f1792a84ab978e1e138be0283f04aa21e839d67f2ed001924010348
```

Runtime: vanilla Ubuntu 26.04.1 LTS, glibc 2.43, OpenSSH 10.2p1, ARM64.
The compiler stage remains the pinned Ubuntu 22.04 stage. The two test binaries
and libcudart are byte-identical to the published Ubuntu 22.04 r2 image:

```text
6628cf01fb242d0f09552c732f10d625048eb4cf372201ea084e92e7d27b4e28  cuda-driver-probe
a43539a0e0044aea67c3956f1623c19a074a89dcf99fdaed50121ccba7cdc7e1  cuda-vector-add
dd6bedeab2dca09b04c626ca24834fefd09ad92edf0d86eb3c78d2f558db9c04  libcudart.so.12
```

The existing local test suite passed for this image: runtime dependencies,
driver/stub/host-key absence, boot failure reporting, root and non-root SSH on
port 22, supplementary GID 28, and graceful shutdown. The `cuda` account is
UID/GID 1001 in this variant;
the device-access group remains numeric GID 28.

User-supplied device output subsequently confirmed Ubuntu 26.04.1 (Admiral
Managed), root and UID/GID 1001 plus supplementary GID 28. All four checks passed:
root smoke, root forced PTX JIT, non-root smoke, and non-root forced PTX JIT.
Each verified all 1,048,579 results with CUDA runtime 12.6 and driver API 13.2.
The injected driver path was `/run/admiral/nvidia/lib/libcuda.so.1`.
The supplied output did not independently show the deployed image digest.
Kernel-log review and repeated recreation/reboot/cold-boot acceptance remain
outstanding. These results validate the tested CUDA probes, not LLM inference.

## Local LLM application

Published `alexturner/workload-local-llm:2026-09-20-qwen3-1.7b-r1` and `latest`:

```text
alexturner/workload-local-llm@sha256:789640cb026870b7cd0d71a78a1db0e951a4f7c8e9786121b1e95c75e4e1e38f
```

Qwen3 1.7B Q4_K_M and llama.cpp v0.4.1 on the pinned Ubuntu 26.04 diagnostic
base, with cuBLAS 12.6.4.1 added. Model included; chat UI/API on port 8080 and
SSH on port 22. Normal startup requires CUDA and runs inference as `cuda`.
Local packaging, SSH, no-GPU failure behavior, and explicitly CPU-only chat
generation checks passed. The user subsequently reported the LLM container
running successfully on the Jetson. Detailed offload/performance logs and restart
results remain to be captured. See [usage](llm/README.md) and [test record](llm/VALIDATION.md).
