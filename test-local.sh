#!/bin/bash
# Run on a Docker host without an injected NVIDIA driver. This verifies image
# plumbing and failure behavior; it cannot validate GPU execution.
set -euo pipefail
image=${1:-alexturner/workload-cuda-debug:2026-09-20-ubuntu26.04-cuda12.6.3-r1}
tmp=$(mktemp -d)
container=
cleanup() {
    if [[ -n $container ]]; then docker rm -f "$container" >/dev/null; fi
    rm -rf "$tmp"
}
trap cleanup EXIT
ssh-keygen -q -t ed25519 -N '' -f "$tmp/test-key"

docker run --rm --entrypoint /bin/bash "$image" -ec '
    test "$(dpkg --print-architecture)" = arm64
    test -z "$(find /etc/ssh -name "ssh_host_*" -print -quit)"
    test -z "$(find /usr -name "libcuda.so*" -print -quit)"
    test -z "$(find /usr -type d -name stubs -print -quit)"
    ldd /usr/local/bin/cuda-vector-add
    ! ldd /usr/local/bin/cuda-vector-add | grep -q "not found"
    id -G cuda | tr " " "\n" | grep -qx 28
    ssh-keygen -lf /root/.ssh/authorized_keys
'
container=$(docker run -d -p 127.0.0.1::22 "$image")
for ((i=0; i<30; i++)); do
    if docker exec "$container" test -f /var/log/cuda-debug/boot.exitcode; then break; fi
    sleep 1
done
docker exec "$container" bash -ec '
    test "$(cat /var/log/cuda-debug/boot.exitcode)" -ne 0
    grep -q "FAIL dlopen(libcuda.so.1)" /var/log/cuda-debug/boot.log
    /usr/sbin/sshd -T | grep -E "^(port|authenticationmethods|passwordauthentication|kbdinteractiveauthentication|permitrootlogin) "
    test "$(/usr/sbin/sshd -T | sed -n "s/^authenticationmethods //p")" = publickey
    test "$(/usr/sbin/sshd -T | sed -n "s/^passwordauthentication //p")" = no
    test "$(/usr/sbin/sshd -T | sed -n "s/^kbdinteractiveauthentication //p")" = no
'
# Add an ephemeral test key to this disposable container, never to the image.
docker exec -i "$container" bash -c 'cat >> /root/.ssh/authorized_keys' < "$tmp/test-key.pub"
docker exec -i "$container" bash -c 'cat >> /home/cuda/.ssh/authorized_keys' < "$tmp/test-key.pub"
port=$(docker port "$container" 22/tcp | awk -F: '{print $NF}')
hostkey=$(docker exec "$container" cat /etc/ssh/ssh_host_ed25519_key.pub)
printf '[127.0.0.1]:%s %s\n' "$port" "$hostkey" > "$tmp/known_hosts"
ssh_options=(-F /dev/null -i "$tmp/test-key" -p "$port" -o IdentitiesOnly=yes
    -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes
    -o "UserKnownHostsFile=$tmp/known_hosts")
ssh "${ssh_options[@]}" root@127.0.0.1 'test "$(id -u)" = 0 && cuda-debug inspect' > "$tmp/inspect.log"
ssh "${ssh_options[@]}" cuda@127.0.0.1 'test "$(id -u)" -ne 0 && id -G | tr " " "\n" | grep -qx 28'
if ssh "${ssh_options[@]}" cuda@127.0.0.1 'cuda-debug smoke' > "$tmp/smoke.log" 2>&1; then
    echo 'FAIL: smoke unexpectedly passed without a GPU driver' >&2
    exit 1
fi
grep -q 'FAIL dlopen(libcuda.so.1)' "$tmp/smoke.log"
docker stop -t 5 "$container" >/dev/null
test "$(docker inspect -f '{{.State.ExitCode}}' "$container")" = 0
echo 'PASS: ARM64 image, runtime linkage, no bundled driver/host keys, boot failure reporting, SSH root/non-root auth, GID 28, graceful stop'
