#!/usr/bin/env bash
set -u

out=/root/wmma-results/profile-blockpack-p8

restore_production() {
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null
}
trap restore_production EXIT INT TERM

sudo -n /usr/local/sbin/ember-cert-production stop
rm -rf "${out}"
mkdir -p "${out}"

timeout --signal=INT --kill-after=30s 300s \
    docker run --rm --name codex-profile-blockpack \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add 105 \
    --ipc=host --security-opt seccomp=unconfined \
    -e LD_LIBRARY_PATH=/opt/rocm/lib \
    -e ROCM_PATH=/opt/rocm/core-7.14 \
    -v /root/wmma-half50-codex:/work -w /work \
    -v /root/wmma-results:/results \
    rocm/dev-ubuntu-24.04:7.14.0-full \
    rocprof-compute profile \
        --output-directory /results/profile-blockpack-p8 \
        -d 1 \
        -b 2.1.3 2.1.8 2.1.9 3.4.3 7.9.2 7.9.3 7.5.0 \
        -- /bin/bash -lc './bp-tune-base 1 1 || true'
