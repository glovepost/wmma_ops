#!/usr/bin/env bash
set -u

for binary in \
    half-waveprivate-2x2-s1 \
    half-waveprivate-2x2-s2 \
    half-waveprivate-2x2-s4 \
    half-waveprivate-2x2-s8 \
    half-waveprivate-2x2-s16 \
    half-waveprivate-2x2-s32 \
    half-waveprivate-4x1-s16 \
    half-waveprivate-1x4-s16; do
    echo "=== ${binary} ==="
    "./${binary}" 20 10
    rc=$?
    if (( rc != 0 && rc != 1 )); then
        exit "${rc}"
    fi
done
