#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Ivendor-waveprivate/include -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=256 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)
llvm=/opt/rocm/llvm/bin

# Build the source image.  Its 119-VGPR descriptor and revised refill schedule
# do not match the leader's guarded hand transforms; forcing those textual
# rewrites would not be a valid composition experiment.
hipcc "${common[@]}" -DWMMA_BP_PAIR_LANE_EPILOGUE=1 --cuda-device-only -S \
    tools/rocwmma_record.hip -o traces/bp-pair-lane-source.s

assemble() {
    local candidate=$1
    "${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
        -mcode-object-version=6 -c "traces/${candidate}.s" \
        -o "traces/${candidate}.o"
    "${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
        -o "traces/${candidate}.out" "traces/${candidate}.o"
    "${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
        -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
        -input=/dev/null -input="traces/${candidate}.out" \
        -output="traces/${candidate}.hipfb"
}

assemble bp-pair-lane-source
hipcc --offload-host-only "${common[@]}" -DWMMA_BP_PAIR_LANE_EPILOGUE=1 \
    -Xclang -fcuda-include-gpubinary \
    -Xclang traces/bp-pair-lane-source.hipfb \
    -c tools/rocwmma_record.hip -o traces/bp-pair-lane-source-host.o
hipcc traces/bp-pair-lane-source-host.o -lrocblas -o bp-pair-lane-source

echo "bp-pair-lane-source:"
grep -E '^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-pair-lane-source.s
printf '  dpp_pack=%s global_store_b32=%s global_store_b16=%s\n' \
    "$(grep -c 'v_pack_b32_f16.*row_xmask' traces/bp-pair-lane-source.s || true)" \
    "$(grep -c 'global_store_b32' traces/bp-pair-lane-source.s || true)" \
    "$(grep -c 'global_store_.*b16' traces/bp-pair-lane-source.s || true)"
