#!/usr/bin/env bash
set -euo pipefail

for f in half-waveprivate-2x2-s16 half-waveprivate-packc-m half-packc-wide-4x8 half-packc-wide-8x4; do
    echo "===${f}==="
    /opt/rocm/llvm/bin/llvm-readelf -n "${f}" \
        | grep -E 'group_segment_fixed_size|private_segment_fixed_size|sgpr_count|vgpr_count'
done
