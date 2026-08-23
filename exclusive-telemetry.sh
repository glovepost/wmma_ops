#!/usr/bin/env bash
set -u

restore_production() {
    sudo -n /usr/local/sbin/ember-cert-production start >/dev/null
}
trap restore_production EXIT INT TERM

sudo -n /usr/local/sbin/ember-cert-production stop
docker run --rm --name codex-wmma-telemetry \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -e LD_LIBRARY_PATH=/opt/rocm/lib \
    -v /root/wmma-half50-codex:/work -w /work \
    wmma-record:rocm714-torch213 ./half-base 200 1000 &
bench_pid=$!

while kill -0 "${bench_pid}" 2>/dev/null; do
    sudo -n awk '/GFX Clocks|GPU Load|Socket Graphics Package Power|Temperature/ {print}' \
        /sys/kernel/debug/dri/1/amdgpu_pm_info 2>/dev/null | tr '\n' ' '
    echo
    sleep 0.1
done

wait "${bench_pid}"
rc=$?
if (( rc != 0 && rc != 1 )); then
    exit "${rc}"
fi
