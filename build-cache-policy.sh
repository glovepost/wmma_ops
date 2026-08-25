#!/usr/bin/env bash
set -euo pipefail

cd /work

# RDNA 3.5's ENC_MUBUF format exposes independent GLC, SLC, and DLC bits.
# Patch only the selected loop's three refill loads, preserving instruction
# count, addresses, register allocation, and the qualified LDS-only wait.
common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=256 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)
llvm=/opt/rocm/llvm/bin
mkdir -p traces

python3 tools/patch_publish_wait_asm.py \
    traces/bp-register-phase-d2.s traces/bp-publish-lgkm0.s 0

for policy in none glc slc dlc glc-slc glc-dlc slc-dlc glc-slc-dlc; do
    candidate="bp-cache-${policy}"
    python3 tools/patch_cache_policy_asm.py \
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
    hipcc --offload-host-only "${common[@]}" \
        -Xclang -fcuda-include-gpubinary \
        -Xclang "traces/${candidate}.hipfb" \
        -c tools/rocwmma_record.hip -o "traces/${candidate}-host.o"
    hipcc "traces/${candidate}-host.o" -lrocblas -o "${candidate}"
    echo "${candidate}:"
    grep -E '^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
        "traces/${candidate}.s"
done
