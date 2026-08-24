#!/usr/bin/env bash
set -u

restore_production() {
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null
}
trap restore_production EXIT INT TERM

sudo -n /usr/local/sbin/ember-cert-production stop
cd /root/wmma-half50-codex

binaries=(
    bp-row-isolation-base
    bp-row-isolation-g1
    bp-row-isolation-g2
    bp-row-isolation-g3
    bp-row-isolation-g12
    bp-row-isolation-g13
    bp-row-isolation-g23
    bp-row-isolation-all
    bp-row-isolation-base
)

for binary in "${binaries[@]}"; do
    echo "=== ${binary} ==="
    timeout --signal=INT --kill-after=10s 90s \
        docker run --rm --name "codex-${binary}" \
        --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
        -e LD_LIBRARY_PATH=/opt/rocm/lib \
        -v /root/wmma-half50-codex:/work -w /work \
        wmma-record:rocm714-torch213 "./${binary}" \
        "${WARMUP:-20}" "${ITERATIONS:-20}"
    result=$?
    if (( result != 0 && result != 1 )); then
        exit "${result}"
    fi
done
