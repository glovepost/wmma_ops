#!/usr/bin/env bash
set -euo pipefail

cd /work
llvm=/opt/rocm/llvm/bin
asm_dir=/work/asm-p8-read2
mkdir -p "${asm_dir}"

common=(--offload-host-only --offload-arch=gfx1151 -O3 -std=c++20
        -I../include-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

build_variant() {
    local candidate=$1
    shift
    python3 ../tools/make-p8-read2.py "$@" ../bp-p4-read2.s "${candidate}.s"
    "${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
        -mcode-object-version=6 -c "${candidate}.s" -o "${candidate}.o"
    "${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
        -o "${candidate}.out" "${candidate}.o"
    "${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
        -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
        -input=/dev/null -input="${candidate}.out" \
        -output="${candidate}.hipfb"
    hipcc "${common[@]}" \
        -Xclang -fcuda-include-gpubinary -Xclang "${candidate}.hipfb" \
        -c ../rocwmma_half_record.hip -o "${candidate}-host.o"
    hipcc "${candidate}-host.o" -lrocblas -o "../${candidate}"
    echo "${candidate}:"
    "${llvm}/llvm-readelf" -n "${candidate}.out" \
        | grep -E 'group_segment_fixed_size|vgpr_count|sgpr_count'
}

cd "${asm_dir}"
build_variant bp-p8-read2
build_variant bp-p8-read2-loads --b128-stores
