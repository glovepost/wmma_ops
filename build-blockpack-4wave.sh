#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=128 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

report_resources() {
    grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):' \
        "$1"
}

for waves_min in 2 3 4; do
    candidate="bp-4wave-p8-wmin${waves_min}"
    hipcc "${common[@]}" -DWMMA_BP_WAVES_MIN="${waves_min}" \
        rocwmma_half_record.hip -lrocblas -o "${candidate}"
    hipcc "${common[@]}" -DWMMA_BP_WAVES_MIN="${waves_min}" \
        --cuda-device-only -S rocwmma_half_record.hip \
        -o "traces/${candidate}.s"
    echo "${candidate}:"
    report_resources "traces/${candidate}.s"
done

candidate="bp-4wave-p8-stream-b"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    rocwmma_half_record.hip -lrocblas -o "${candidate}"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    --cuda-device-only -S rocwmma_half_record.hip \
    -o "traces/${candidate}.s"
echo "${candidate}:"
report_resources "traces/${candidate}.s"

candidate="bp-4wave-p8-stream-b-late-refill"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    -DWMMA_BP_LATE_B_REFILL=1 \
    rocwmma_half_record.hip -lrocblas -o "${candidate}"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    -DWMMA_BP_LATE_B_REFILL=1 \
    --cuda-device-only -S rocwmma_half_record.hip \
    -o "traces/${candidate}.s"
echo "${candidate}:"
report_resources "traces/${candidate}.s"

candidate="bp-4wave-p8-stream-b-late-refill-mubuf"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    -DWMMA_BP_LATE_B_REFILL=1 -DWMMA_BP_BUFFER_A_PREFETCH=1 \
    rocwmma_half_record.hip -lrocblas -o "${candidate}"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    -DWMMA_BP_LATE_B_REFILL=1 -DWMMA_BP_BUFFER_A_PREFETCH=1 \
    --cuda-device-only -S rocwmma_half_record.hip \
    -o "traces/${candidate}.s"
echo "${candidate}:"
report_resources "traces/${candidate}.s"

candidate="bp-4wave-p8-stream-b-late-b1"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    -DWMMA_BP_LATE_B1_PREFETCH=1 \
    rocwmma_half_record.hip -lrocblas -o "${candidate}"
hipcc "${common[@]}" -DWMMA_BP_STREAM_B_BARRIER=1 \
    -DWMMA_BP_LATE_B1_PREFETCH=1 \
    --cuda-device-only -S rocwmma_half_record.hip \
    -o "traces/${candidate}.s"
echo "${candidate}:"
report_resources "traces/${candidate}.s"
