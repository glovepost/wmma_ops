#!/usr/bin/env bash
set -u

restore_production() {
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null
}
trap restore_production EXIT INT TERM

sudo -n /usr/local/sbin/ember-cert-production stop
cd /root/wmma-half50-codex

binaries=("$@")
if (( ${#binaries[@]} == 0 )); then
    binaries=(bench-wmma-iu4-gemm)
fi

for binary in "${binaries[@]}"; do
    echo "=== ${binary} ==="
    timeout --signal=INT --kill-after=10s 120s \
    docker run --rm --name codex-wmma-iu4-gemm \
        --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
        -e LD_LIBRARY_PATH=/opt/rocm/lib \
        -v /root/wmma-half50-codex:/work -w /work \
        wmma-record:rocm714-torch213 "./${binary}" \
        "${SIZE:-4096}" "${WARMUP:-20}" "${SAMPLES:-20}"
    rc=$?
    if (( rc != 0 && rc != 1 )); then
        exit "${rc}"
    fi
done
