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

build_host() {
    local candidate=$1
    shift
    hipcc "${common[@]}" "$@" \
        -Xclang -fcuda-include-gpubinary \
        -Xclang traces/bp-publish-lgkm0.hipfb \
        -c tools/rocwmma_record.hip -o "traces/${candidate}-host.o"
    hipcc "traces/${candidate}-host.o" -lrocblas -o "${candidate}"
}

# Rebuild the square host around the same selected device image to prove that
# rectangular bookkeeping does not change the established default contract.
build_host bp-tail-grid512-control
build_host bp-tail-grid480-m -DRECORD_M=3840 -DRECORD_N=4096 -DRECORD_K=4096
build_host bp-tail-grid480-n -DRECORD_M=4096 -DRECORD_N=3840 -DRECORD_K=4096
