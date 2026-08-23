#!/usr/bin/env bash
set -u

restore_production() {
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null
}
trap restore_production EXIT INT TERM

sudo -n /usr/local/sbin/ember-cert-production stop
cd /root/wmma-half50-codex

run_case() {
    local label=$1
    shift
    echo "=== ${label} ==="
    timeout --signal=INT --kill-after=10s 90s \
        docker run --rm --name "codex-offset-${label}" \
        --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
        -e LD_LIBRARY_PATH=/opt/rocm/lib \
        -v /root/wmma-half50-codex:/work -w /work \
        wmma-record:rocm714-torch213 ./bp-tune-base \
        "${WARMUP:-5}" "${ITERATIONS:-3}" "$@"
    local result=$?
    if (( result != 0 && result != 1 )); then
        exit "${result}"
    fi
}

run_case separate-0 0 0 0
for offset in 32 64 128 256 512 1024; do
    run_case "separate-b${offset}" 0 "${offset}" 0
done
for gap in 0 32 64 128 256 512 1024; do
    run_case "combined-gap${gap}" 0 "${gap}" 0 combined-ab
done
