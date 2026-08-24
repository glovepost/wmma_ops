#!/usr/bin/env bash
set -u

restore_production() {
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null
}
trap restore_production EXIT INT TERM

sudo -n /usr/local/sbin/ember-cert-production stop
cd /root/wmma-half50-codex

for run in 1 2 3 4 5; do
    for binary in bp-row-isolation-base bp-row-isolation-all; do
        echo "=== run ${run} ${binary} ==="
        timeout --signal=INT --kill-after=10s 90s \
            docker run --rm --name "codex-row0312-${run}-${binary##*-}" \
            --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
            -e LD_LIBRARY_PATH=/opt/rocm/lib \
            -v /root/wmma-half50-codex:/work -w /work \
            wmma-record:rocm714-torch213 "./${binary}" \
            "${WARMUP:-20}" "${ITERATIONS:-100}"
        result=$?
        if (( result != 0 && result != 1 )); then
            exit "${result}"
        fi
    done
done
