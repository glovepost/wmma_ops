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
symbol=_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii

python3 tools/patch_publish_wait_asm.py \
    traces/bp-register-phase-d2.s traces/bp-publish-lgkm0.s 0
# Rebuild the selected control code object as a standalone module too. Both
# halves must differ by exactly the one entry SALU instruction above.
"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c traces/bp-publish-lgkm0.s \
    -o traces/bp-split-base.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o traces/bp-split-base.out traces/bp-split-base.o

build_host() {
    local candidate=$1
    shift
    hipcc "${common[@]}" "$@" \
        -Xclang -fcuda-include-gpubinary \
        -Xclang traces/bp-publish-lgkm0.hipfb \
        -c tools/rocwmma_record.hip -o "traces/${candidate}-host.o"
    hipcc "traces/${candidate}-host.o" -lrocblas -o "${candidate}"
}

build_host bp-split-control
build_split() {
    local first=$1
    local second=$((512 - first))
    local mode=$2
    local candidate="bp-split-${mode}-${first}-${second}"
    local offset="traces/bp-split-offset${first}"
    python3 tools/offset_workgroup_id_asm.py \
        traces/bp-publish-lgkm0.s "${offset}.s" "${first}"
    "${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
        -mcode-object-version=6 -c "${offset}.s" -o "${offset}.o"
    "${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
        -o "${offset}.out" "${offset}.o"
    local split_common=(-DRECORD_SPLIT_MODULE_DISPATCH=1
                        "-DRECORD_GRID_BLOCKS=${first}"
                        "-DRECORD_GRID_BLOCKS_B=${second}"
                        '-DRECORD_MODULE0_PATH="traces/bp-split-base.out"'
                        "-DRECORD_MODULE1_PATH=\"${offset}.out\"")
    if [[ "${mode}" == concurrent ]]; then
        split_common+=(-DRECORD_SPLIT_CONCURRENT=1)
    fi
    build_host "${candidate}" "${split_common[@]}"
}

build_split 256 serial
build_split 240 concurrent
build_split 256 concurrent
build_split 280 concurrent
build_split 320 concurrent

for image in traces/bp-split-base.out traces/bp-split-offset{240,256,280,320}.out; do
    "${llvm}/llvm-readelf" --notes "${image}" | \
        grep -E "(\.name: ${symbol}|\.group_segment_fixed_size:|\.sgpr_count:|\.vgpr_count:)"
done
