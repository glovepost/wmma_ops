#!/usr/bin/env bash
set -u

restore_production() {
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null
}
trap restore_production EXIT INT TERM

sudo -n /usr/local/sbin/ember-cert-production stop

for binary in half-base half-single-packc-w16; do
    echo "=== ${binary} ==="
    docker run --rm --name "codex-exclusive-${binary}" \
        --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
        -e LD_LIBRARY_PATH=/opt/rocm/lib \
        -v /root/wmma-half50-codex:/work -w /work \
        wmma-record:rocm714-torch213 "./${binary}" 40 20
    rc=$?
    if (( rc != 0 && rc != 1 )); then
        exit "${rc}"
    fi
done
