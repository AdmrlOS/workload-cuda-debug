# Admiral CUDA debug

ARM64 Ubuntu diagnostic userspace for Jetson Orin Nano Super. Workload name:
`cuda-debug`. The current Dockerfile defaults to an **Ubuntu 26.04 proof of concept**,
with the same CUDA 12.6.3 / libcudart 12.6.77 from NVIDIA's Jetson ARM64 repository.
Both the Ubuntu 26.04 variant and Ubuntu 22.04 `r2` image passed root and non-root
execution and forced PTX JIT in user-supplied device logs.
Recreation, reboot/cold-boot repetition and kernel-log review remain outstanding.
See [PUBLISHED.md](PUBLISHED.md) for image digests and validation scope.
For the local LLM application image, see [llm/README.md](llm/README.md).

The final stage uses vanilla Ubuntu 26.04 and its own glibc and system packages.
The compiler stage remains Ubuntu 22.04; only the compiled test binaries, CUDA
runtime, license and build records are copied forward. This tests running the same
CUDA application on a newer userspace, not compiling CUDA 12.6 with Ubuntu 26.04's
toolchain. No NVIDIA apt repository is installed in the final image.
Both bases are pinned by digest and the CUDA compiler/runtime packages by version.
Ubuntu diagnostic package versions are recorded inside the image; their apt
repositories track updates, so use the published image digest for exact reproduction.

The final image contains no NVIDIA driver, CUDA stub, compiler, container toolkit,
or Docker daemon. Admiral supplies the driver and devices. About 12 MiB of device
memory is used by the vector-add test, plus CUDA context overhead and host buffers.
The compiler exists only in the build stage. Sources are included under
`/opt/cuda-debug/src`, along with build information and package inventories.

## Build and publish

```sh
docker buildx build --platform linux/arm64 --load \
  -t alexturner/workload-cuda-debug:2026-09-20-ubuntu26.04-cuda12.6.3-r1 \
  -t alexturner/workload-cuda-debug:ubuntu26.04 .
bash test-local.sh
docker push alexturner/workload-cuda-debug:2026-09-20-ubuntu26.04-cuda12.6.3-r1
docker push alexturner/workload-cuda-debug:ubuntu26.04
```

`latest` stays on the validated Ubuntu 22.04 `2026-09-20-cuda12.6.3-r2` image.
Use the explicit Ubuntu 26.04 tag when deploying this experiment.
To build the current sources against the older runtime base without replacing
the original published image:

```sh
docker buildx build --platform linux/arm64 --load \
  --build-arg UBUNTU_IMAGE=ubuntu:22.04@sha256:1cc7bb38a74c0e126716646e47c0b3c5c139547d386d5ff7a64cf3ea316ca523 \
  --build-arg IMAGE_VERSION=ubuntu22.04-local \
  -t workload-cuda-debug:ubuntu22.04-local .
```

`ssh/authorized_keys` is a checked-in snapshot of
https://github.com/alexanderturner.keys fetched on 2026-09-20:
`SHA256:yp9F51C0Pp4QoSw3WbEKDpSsnGoh0n0UXjK/p6qqCUE` (ED25519).
To rotate access, update that file and rebuild. Startup never downloads keys.

## Deploy

Deploy the image through the Admiral portal/cloud as workload `cuda-debug`, with
`IsolateDevices=false` and TCP port 22 reachable. Leave the image entrypoint
intact. Set no NVIDIA container-toolkit variables and install no extra drivers.
Admiral's expected profile is `tegra234-nvgpu-r39.2.1-v1`. Do not mount over `/run`,
`/usr/lib`, or other driver mount parents.

```sh
ssh -p 22 root@DEVICE_IP
cuda-debug inspect
cuda-debug smoke
cuda-debug jit
cat /var/log/cuda-debug/boot.exitcode
cat /var/log/cuda-debug/boot.log
```

Both `root` and `cuda` accept only the included GitHub key, with passwords and
keyboard-interactive authentication disabled. `cuda` has supplementary numeric
GID 28 for Admiral's video devices and no sudo access. Test that path separately:

```sh
ssh -p 22 cuda@DEVICE_IP 'id; cuda-debug smoke'
# Or from a root session:
runuser -u cuda -- cuda-debug smoke
runuser -u cuda -- cuda-debug jit
```

Host keys are generated at first startup and retained for that writable userspace.
Recreating the userspace generates new keys unless `/etc/ssh` is persisted through
a suitable Admiral configuration; verify the new fingerprints before replacing
your known-hosts entry. Do not disable SSH host-key checking.

SSH remains running if the automatic smoke test fails. The boot test has a
60-second timeout (plus a five-second kill grace), controlled by
`CUDA_SMOKE_TIMEOUT`; set `CUDA_SMOKE_ON_BOOT=0` to disable it. A failed test is
logged and recorded as a nonzero `boot.exitcode`; it is never reported as a pass.
Manual smoke tests have no built-in timeout; use `timeout 60 cuda-debug smoke`.

## What the checks prove

`inspect` collects identity, manifest, devices, mounts, loader cache, dependencies,
and memory without requiring a functioning GPU. `driver` dynamically loads
`libcuda.so.1`, prints the resolved library path and driver API version, calls
`cuInit`, enumerates devices, and creates/releases a primary context on each.
`smoke` then uses the shared CUDA runtime to allocate buffers, copy input, launch
vector addition, synchronize, copy results back, and verify all 1,048,579 outputs
on every visible GPU. It exits nonzero on any CUDA error or incorrect result.
There is no CPU fallback. The binary includes native `sm_87` code and `compute_87`
PTX; `jit` forces the optional PTX path without using a cached compiled kernel.

For loader or syscall debugging:

```sh
LD_DEBUG=libs cuda-debug driver 2>&1 | tee /tmp/cuda-loader.log
strace -f -o /tmp/cuda-strace.log cuda-debug smoke
dmesg > /tmp/kernel-before.log
cuda-debug smoke
dmesg > /tmp/kernel-after.log
diff -u /tmp/kernel-before.log /tmp/kernel-after.log
```

Kernel logs and ptrace depend on the platform's actual capabilities/policy; do not
change device permissions or reload host GPU modules to work around errors.
`nvidia-smi`, display, graphics, multimedia and profiling are outside this image's
acceptance scope. SSH commands use normal driver library bindings; `cuda-debug`
also prepends `/run/admiral/nvidia/lib` if present.

Record the deployed image digest, Admiral version, host manifest/profile, driver
and runtime versions, process UID/groups, test output, and kernel log delta.
Repeat root and non-root tests after userspace recreation, reboot, and cold boot.
A local build or a successful SSH login does not establish GPU acceptance.

References: [NVIDIA JetPack 6.2.1 release notes](https://docs.nvidia.com/jetson/jetpack/release-notes/)
(CUDA 12.6 on Orin), [Jetson ARM64 package repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/).
The [CUDA 12.6.3 installation guide](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html)
documents its compiler environments. Our Ubuntu 26.04 runtime experiment is not
a claim of NVIDIA support for that OS/toolkit combination.
