#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=256 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

hipcc "${common[@]}" --cuda-device-only -S rocwmma_half_record.hip \
    -o traces/bp-row-isolation-source.s
python3 tools/patch_buffer_prefetch_asm.py \
    traces/bp-row-isolation-source.s traces/bp-row-isolation-soffset.s --scalar-offset
python3 tools/patch_progressive_commit_asm.py \
    traces/bp-row-isolation-soffset.s traces/bp-row-isolation-progressive.s
python3 tools/shift_vgpr_boundary_asm.py \
    traces/bp-row-isolation-progressive.s traces/bp-row-isolation-base.s 2

declare -A schedules=(
    [base]="0123 0123 0123"
    [g1]="0312 0123 0123"
    [g2]="0123 0312 0123"
    [g3]="0123 0123 0312"
    [g12]="0312 0312 0123"
    [g13]="0312 0123 0312"
    [g23]="0123 0312 0312"
    [all]="0312 0312 0312"
)

for name in "${!schedules[@]}"; do
    read -r order1 order2 order3 <<< "${schedules[${name}]}"
    python3 tools/reorder_hot_wmma_rows_asm.py \
        traces/bp-row-isolation-base.s "traces/bp-row-isolation-${name}.s" \
        "${order1}" "${order2}" "${order3}"
done

llvm=/opt/rocm/llvm/bin
build_candidate() {
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
    hipcc --offload-host-only "${common[@]}" \
        -Xclang -fcuda-include-gpubinary \
        -Xclang "traces/${candidate}.hipfb" \
        -c rocwmma_half_record.hip -o "traces/${candidate}-host.o"
    hipcc "traces/${candidate}-host.o" -lrocblas -o "${candidate}"
    echo "${candidate}:"
    grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):' \
        "traces/${candidate}.s"
}

for name in "${!schedules[@]}"; do
    build_candidate "bp-row-isolation-${name}"
done
