#!/bin/bash
# CPU inference here verifies packaging/API only; GPU inference requires Orin.
set -euo pipefail
image=${1:-alexturner/workload-local-llm:2026-09-20-qwen3-1.7b-r1}
tmp=$(mktemp -d)
production=
cpu=
cleanup() {
    [[ -z $production ]] || docker rm -f "$production" >/dev/null
    [[ -z $cpu ]] || docker rm -f "$cpu" >/dev/null
    rm -rf "$tmp"
}
trap cleanup EXIT
docker run --rm --entrypoint /bin/bash "$image" -ec '
    test "$(dpkg --print-architecture)" = arm64
    test -z "$(find /usr -name "libcuda.so*" -print -quit)"
    test -z "$(find /usr -type d -name stubs -print -quit)"
    test -z "$(find /etc/ssh -name "ssh_host_*" -print -quit)"
    ! ldd /opt/llama/bin/llama-server /opt/llama/bin/libggml-cuda.so | grep -q "not found"
    echo "d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5  /opt/llama/models/Qwen3-1.7B-Q4_K_M.gguf" | sha256sum -c -
    id -G cuda | tr " " "\n" | grep -qx 28
'

production=$(docker run -d -p 127.0.0.1::22 "$image")
for ((i=0; i<30; i++)); do
    if docker exec "$production" test -f /var/log/local-llm/server.exitcode; then break; fi
    sleep 1
done
docker exec "$production" bash -ec '
    test "$(cat /var/log/local-llm/server.exitcode)" -ne 0
    grep -q "FAIL dlopen(libcuda.so.1)" /var/log/local-llm/server.log
    ! pgrep -x llama-server
    /usr/sbin/sshd -t
'
ssh-keygen -q -t ed25519 -N '' -f "$tmp/key"
docker exec -i "$production" bash -c 'cat >> /root/.ssh/authorized_keys' < "$tmp/key.pub"
docker exec -i "$production" bash -c 'cat >> /home/cuda/.ssh/authorized_keys' < "$tmp/key.pub"
ssh_port=$(docker port "$production" 22/tcp | awk -F: '{print $NF}')
hostkey=$(docker exec "$production" cat /etc/ssh/ssh_host_ed25519_key.pub)
printf '[127.0.0.1]:%s %s\n' "$ssh_port" "$hostkey" > "$tmp/known_hosts"
ssh_options=(-F /dev/null -i "$tmp/key" -p "$ssh_port" -o IdentitiesOnly=yes
    -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes
    -o "UserKnownHostsFile=$tmp/known_hosts")
ssh "${ssh_options[@]}" root@127.0.0.1 'test "$(id -u)" = 0'
ssh "${ssh_options[@]}" cuda@127.0.0.1 'test "$(id -u)" -ne 0 && id -G | tr " " "\n" | grep -qx 28'
docker stop -t 5 "$production" >/dev/null
test "$(docker inspect -f '{{.State.ExitCode}}' "$production")" = 0
echo 'PASS: missing GPU prevents inference while root/non-root SSH stays available'

# Explicitly bypass production startup ONLY in this disposable CPU test.
cpu=$(docker run -d --user cuda -p 127.0.0.1::8080 \
    --entrypoint /opt/llama/bin/llama-server "$image" \
    --model /opt/llama/models/Qwen3-1.7B-Q4_K_M.gguf --alias qwen3-1.7b \
    --offline --device none --n-gpu-layers 0 --fit off \
    --host 0.0.0.0 --port 8080 --ctx-size 512 --parallel 1 --threads 4 \
    --batch-size 128 --ubatch-size 128 --flash-attn off \
    --jinja --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0)
port=$(docker port "$cpu" 8080/tcp | awk -F: '{print $NF}')
ready=0
for ((i=0; i<120; i++)); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" > "$tmp/health.json" 2>/dev/null; then ready=1; break; fi
    if [[ $(docker inspect -f '{{.State.Running}}' "$cpu") != true ]]; then break; fi
    sleep 1
done
if [[ $ready != 1 ]]; then docker logs "$cpu"; exit 1; fi
curl -fsS --compressed "http://127.0.0.1:$port/" > "$tmp/index.html"
grep -qi '<html' "$tmp/index.html"
curl -fsS --max-time 180 "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":24,"temperature":0,"stream":false}' \
    > "$tmp/response.json"
python3 - "$tmp/response.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
content=r['choices'][0]['message']['content']
assert content and content.strip(), r
assert r['usage']['completion_tokens'] > 0, r
print('CPU packaging check response:', content)
print('PASS: model loads, bundled chat HTML is served, chat API generates tokens')
PY
docker stop -t 10 "$cpu" >/dev/null
test "$(docker inspect -f '{{.State.ExitCode}}' "$cpu")" = 0
echo 'PASS: local checks complete; CUDA LLM execution is still unverified'
