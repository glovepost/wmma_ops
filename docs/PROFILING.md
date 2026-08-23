# Profiling WMMA kernels on gfx1151

This guide is for AMD Strix Halo (`gfx1151`) and current ROCm 7.x tools. Its
main rule is simple: **timing and counter collection are different passes**.
Counter collection can serialize dispatches, so a duration reported by a PMC
run is not valid input to a TFLOPS calculation.

Command syntax and gfx1151 support links were reviewed against ROCm 7.14-era
documentation on 2026-08-23. Always consult the installed tool's `--help`
because profiler interfaces and counter names continue to evolve.

For the active target, schedule, and record gate, see
[the performance ledger](PERFORMANCE_STATUS.md).

## Before touching the GPU

Capture enough context to make the result interpretable:

```bash
git rev-parse HEAD
git status --short
/opt/rocm/bin/hipcc --version
rocminfo | rg -n 'Name:|Marketing Name:|Wavefront Size:|Max Waves Per CU'
rocm-smi --showproductname --showdriverversion --showclocks \
  --showpower --showtemp
```

On a shared Strix Halo host, also confirm that no model server or stale process
owns most of unified memory. A second model loader may silently change device
placement and invalidate both correctness and timing. Use the host's advisory
GPU lock for any command that loads a model or profiles the GPU.

Keep these three classes of experiment separate:

| Pass | Purpose | Valid outputs |
|---|---|---|
| Correctness | Compare complete FP32 output with a trusted reference | Error metrics and pass/fail |
| Timing | Warm, then time without profiler counters or telemetry polling | Kernel/API latency and TFLOPS |
| Diagnostic | Collect counters, traces, clocks, power, or ISA | Bottleneck evidence, not record timing |

## Reproduce the record candidate first

Build and run the pinned standalone harness before profiling a changed kernel:

```bash
git clone https://github.com/adelj88/rocm_wmma_gemm.git /tmp/rocm_wmma_gemm
git -C /tmp/rocm_wmma_gemm checkout \
  281b5dfd7fbff9cea80753bc55274a54f4a7c53a
tools/build_rocwmma_record.sh /tmp/rocm_wmma_gemm build/rocwmma_record
build/rocwmma_record 200 100 0 2048 0 combined-ab
```

This establishes whether the host is in the same performance regime as the
published audit. Do not debug a candidate against a slow or contaminated
baseline.

## Timing pass

The standalone harness performs five HIP-event timing blocks after warm-up and
then validates the full result. For extension-level measurements, use the JSON
benchmark:

```bash
python3 benchmark_record.py \
  --warmup 10 --iterations 100 --blocks 5 \
  --run-id 1 --output runs/record-1.json
```

Run it from five fresh processes with distinct run IDs. The candidate order is
shuffled within each process to reduce fixed ordering bias. Remember that the
extension wrappers allocate FP32 output on every call; the standalone harness
uses preallocated output, so the two measurements answer different questions.

## rocprofv3 trace pass

ROCm tool flags have changed across releases. Check the installed binary before
copying a command from an older guide:

```bash
rocprofv3 --version
rocprofv3 --help
rocprofv3 --list-avail
```

A kernel/HIP trace on current ROCm 7.x follows this shape:

```bash
mkdir -p traces/record-trace
rocprofv3 --output-format csv \
  -d traces/record-trace -o record \
  --kernel-trace --hip-trace \
  -- build/rocwmma_record 20 10 0 2048 0 combined-ab
```

Use a short diagnostic workload. Trace overhead changes launch timing, and a
large record workload creates an unnecessarily large dispatch file.

## Performance-counter pass

Never assume a counter name is portable between ROCm releases or GPU families.
Start with the target's advertised set:

```bash
rocprofv3 --list-avail > traces/available-counters.txt
```

Then collect one counter per pass unless the installed tool proves that the
requested group fits. On the audited gfx1151 stack, requesting multiple
counters together could exceed hardware collection capability and rocprofv3
could terminate badly. The command shape is:

```bash
mkdir -p traces/record-fetch
rocprofv3 --output-format csv \
  -d traces/record-fetch -o fetch \
  --pmc FETCH_SIZE --kernel-trace \
  -- build/rocwmma_record 20 10 0 2048 0 combined-ab
```

Repeat for each available counter. Record unavailable counters instead of
substituting a similarly named counter from another architecture.

The current leader's separate diagnostic pass reported roughly 21.5% of wave
cycles waiting at barriers, 11.2% waiting on LDS instructions, and 59.5%
occupancy. Those values nominate synchronization/LDS scheduling for further
work; they do not turn the profiled duration into a valid TFLOPS result.

### Counter-unit trap

ROCm 7.x has reported `FETCH_SIZE` and `WRITE_SIZE` in kilobytes on the audited
stack. Confirm the unit for the installed release before computing bandwidth.
A mistaken byte/kilobyte assumption changes the answer by 1024x.

## ROCm Compute Profiler

Current ROCm Compute Profiler documentation lists gfx1151 as a supported
accelerator. Availability still depends on the installed driver/tool package.
Discover the local interface rather than relying on an old container:

```bash
rocprof-compute --version
rocprof-compute --help
rocprof-compute profile --help
rocprof-compute analyze --help
```

Use it for occupancy, instruction mix, cache, and LDS investigations. Keep its
timings in the diagnostic category.

## Inspect generated ISA and resources

Source shape is not enough. Always inspect the code object that actually ran.
The PyTorch build uses `-save-temps`; standalone binaries can be inspected with
the LLVM tools from the same ROCm installation:

```bash
/opt/rocm/llvm/bin/llvm-objdump --offloading build/rocwmma_record
/opt/rocm/llvm/bin/llvm-objdump --disassemble --mcpu=gfx1151 \
  build/rocwmma_record > traces/rocwmma-record.s
/opt/rocm/llvm/bin/llvm-readelf --notes build/rocwmma_record \
  > traces/rocwmma-record.notes
```

Record at least:

- VGPR and SGPR counts;
- group-segment/LDS bytes;
- private-segment/scratch bytes;
- wave mode and work-group size;
- count and ordering of `v_wmma_f32_16x16x16_f16`;
- work-group barriers and relevant `s_waitcnt` instructions;
- LDS loads/stores and any DPP or permlane operations.

The validated standalone kernel compiled to 188 VGPRs, 30 SGPRs, 25,344 bytes
of LDS, and zero scratch with the pinned ROCm 7.14 flags. A rebuild that changes
those resources is a different generated candidate even if the C++ template
arguments match.

`analyze_isa.sh` is a historical helper for the PyTorch extension. Review its
paths and assumptions before use; do not treat its textual estimates as
profiler measurements.

## Interpreting a profile

Ask one question at a time:

- High barrier wait with resident waves available: reduce synchronization or
  increase useful output work per barrier.
- High LDS wait: inspect bank mapping, address generation, and the load/store
  schedule before adding more buffering.
- Scratch use: reduce live state first; a source optimization that spills is
  normally a regression.
- Low occupancy with large VGPR/LDS use: compare against the measured latency
  benefit before forcing occupancy upward.
- High cache misses or memory traffic: verify the matrix allocation phase and
  tile traversal; on this APU, A/B placement changed short screens.
- Declining throughput after a very long warm-up: collect package clock/power
  telemetry in a separate pass. The 1000-launch experiment exposed a slower
  sustained power window rather than improving warm-up quality.

Nominal roofline arithmetic is useful only as a hypothesis. At 59.4 TFLOPS and
256 GB/s, the ridge point is about 232 FLOP/byte, but both the peak clock and
sustained memory bandwidth must be measured for a real roofline.

## Legacy repository tooling

`docker/Dockerfile.profiling`, `docker/docker-compose.profiling.yml`,
`run_rocprof.sh`, `rocprof_wmma.py`, and `profile_wmma.py` preserve the original
ROCm 6.3/7.9 workflow. The compose file refers to profiling helper mounts that
are not present in this repository, and its comments predate current gfx1151
counter support. Treat it as historical source material, not a turnkey current
profiling environment.

## Troubleshooting

### No GPU or permission denied

Check `/dev/kfd`, `/dev/dri`, group membership, and container device mappings.
Do not add `HSA_OVERRIDE_GFX_VERSION` by habit; current gfx1151-capable ROCm
stacks should identify the device directly.

### Counter unavailable

Save `rocprofv3 --list-avail`, reduce to one counter per pass, and record the
missing metric. Counter names from CDNA examples are not necessarily available
on RDNA3.5.

### Correctness changes only under load

First rule out another large process consuming unified memory. On the shared
box, a second model loader caused hybrid placement and a false regression.

### Timing varies between fresh processes

Record process-level distributions, allocation offsets, clocks, temperature,
and package power. Do not choose the best process and call it the record.

## References

- [rocprofiler-sdk documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [ROCm Compute Profiler documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)
- [ROCm Compute Profiler compatible accelerators](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/reference/compatible-accelerators.html)
- [LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html)
- [Repository architecture reference audit](wmma_references.md)
