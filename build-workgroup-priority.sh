#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(--offload-host-only --offload-arch=gfx1151 -O3 -std=c++20
        -Ivendor-waveprivate/include -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=256 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)
llvm=/opt/rocm/llvm/bin

python3 tools/patch_publish_wait_asm.py \
    traces/bp-register-phase-d2.s traces/bp-publish-lgkm0.s 0

build_candidate() {
    local policy=$1
    local candidate="bp-priority-${policy}"
    python3 tools/asymmetric_workgroup_priority_asm.py \
        traces/bp-publish-lgkm0.s "traces/${candidate}.s" "${policy}"
    "${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
        -mcode-object-version=6 -c "traces/${candidate}.s" \
        -o "traces/${candidate}.o"
    "${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
        -o "traces/${candidate}.out" "traces/${candidate}.o"
    "${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
        -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
        -input=/dev/null -input="traces/${candidate}.out" \
        -output="traces/${candidate}.hipfb"
    hipcc "${common[@]}" \
        -Xclang -fcuda-include-gpubinary \
        -Xclang "traces/${candidate}.hipfb" \
        -c tools/rocwmma_record.hip -o "traces/${candidate}-host.o"
    hipcc "traces/${candidate}-host.o" -lrocblas -o "${candidate}"

    echo "${candidate}:"
    grep -E '^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
        "traces/${candidate}.s"
}

build_candidate nop
build_candidate even-high
build_candidate odd-high
