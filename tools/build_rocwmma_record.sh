#!/usr/bin/env bash
set -euo pipefail

readonly PINNED_COMMIT=281b5dfd7fbff9cea80753bc55274a54f4a7c53a
readonly PINNED_INCLUDE_SHA256=aa6cd06871d88871ffc1ce47b0391a2e39797099070ee02afc6541a93513911c
readonly SOURCE_ROOT=${1:-}
readonly OUTPUT=${2:-build/rocwmma_record}
readonly ROCM_ROOT=${ROCM_PATH:-/opt/rocm}

if [[ -z "$SOURCE_ROOT" ]]; then
    echo "usage: $0 /path/to/rocm_wmma_gemm [output]" >&2
    exit 2
fi
if [[ ! -f "$SOURCE_ROOT/rocm_wmma_gemm/include/rocm_wmma_gemm/kernel/kernel.hpp" ]]; then
    echo "not a rocm_wmma_gemm checkout: $SOURCE_ROOT" >&2
    exit 2
fi

if command -v git >/dev/null 2>&1; then
    actual_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
    if [[ "$actual_commit" != "$PINNED_COMMIT" ]]; then
        echo "expected rocm_wmma_gemm $PINNED_COMMIT, found $actual_commit" >&2
        exit 2
    fi
fi

actual_include_sha256=$(
    cd "$SOURCE_ROOT"
    LC_ALL=C find rocm_wmma_gemm/include -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | cut -d ' ' -f 1
)
if [[ "$actual_include_sha256" != "$PINNED_INCLUDE_SHA256" ]]; then
    echo "expected pinned include tree $PINNED_INCLUDE_SHA256, found $actual_include_sha256" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
"$ROCM_ROOT/bin/hipcc" \
    --offload-arch=gfx1151 \
    -mcumode \
    -O3 \
    -ffast-math \
    -mllvm -amdgpu-unroll-threshold-local=700 \
    -std=c++20 \
    -I"$SOURCE_ROOT/rocm_wmma_gemm/include" \
    tools/rocwmma_record.hip \
    -lrocblas \
    -o "$OUTPUT"
echo "built $OUTPUT"
