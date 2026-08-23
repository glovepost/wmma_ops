# Repository Guidelines

## Project Structure & Module Organization
Core HIP/C++ kernels live in `wmma_gemm.hip` and the supporting headers `wmma_kernel_variants.hpp`, `wmma_kernel_largetile.hpp`, `wmma_tile_selection.hpp`, and `wmma_device_helpers.hpp`. The custom rocWMMA header patch is in `rocwmma_patch/rocwmma_gfx1151.hpp`. Python entry points and benchmarking utilities are top-level scripts (for example, `autotune.py`, `benchmark_record.py`, and `rocprof_wmma.py`). Use `docs/PERFORMANCE_STATUS.md` for current performance claims and `docs/PROFILING.md` for measurement procedure. `docs/WMMA_DEVELOPMENT_NOTES.md` is a historical notebook; external reference projects are under `examples/`.

## Build, Test, and Development Commands

### Using Docker (Recommended)
The project includes a historical Docker environment with ROCm, PyTorch, and dependencies pre-configured for gfx1151. It mixes ROCm 7.9 Python packages with ROCm 6.3 APT tooling, so do not use it to promote a current record.

```bash
# Build the Docker image (first time only)
docker compose -f docker/docker-compose.benchmark.yml build

# Run the full test suite
docker compose -f docker/docker-compose.benchmark.yml run --rm benchmark \
  bash -c "export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:\$LD_LIBRARY_PATH && \
           cd /workspace/wmma_ops && python test_rocwmma_patch.py"

# Interactive development shell
docker compose -f docker/docker-compose.benchmark.yml run --rm benchmark bash
# Then inside the container:
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
cd /workspace/wmma_ops
pip install -e . --no-build-isolation
python test_rocwmma_patch.py
```

### Host Installation (requires ROCm + hipcc on PATH)
- `python3 -m pip install -e . --no-build-isolation` — build and install the extension in editable mode.
- `./build_and_test.sh` — end-to-end build + import check + test run.
- `python3 test_rocwmma_patch.py` — main correctness/performance test suite. Its historical variant list is not exhaustive; in particular, verify coverage before assuming a newly bound kernel such as `matmul_opt` was exercised.
- `python3 test_fragment_loading.py` — focused fragment-loading correctness checks.
- `python3 autotune.py --quick` — quick tuning run without Optuna.
- `python3 rocprof_wmma.py` — profiling run with rocprof helpers.
- `python3 benchmark_record.py --iterations 100 --run-id 1 --output runs/record-1.json` — record-shape correctness and timing run. Repeat from five fresh processes; do not collect counters in the timing process.
- `tools/build_rocwmma_record.sh /path/to/pinned/rocm_wmma_gemm build/rocwmma_record` — build the standalone validated candidate after following the pinned-source instructions in the README.

## Coding Style & Naming Conventions
Use 4-space indentation for both HIP/C++ and Python. Follow existing naming patterns: snake_case for functions/variables (for example, `wmma_gemm_kernel`), `wmma_*.hpp` for headers, and `test_*.py` for test scripts. There is no enforced formatter; keep changes consistent with adjacent code blocks and avoid style-only diffs.

## Testing Guidelines
Tests run as standalone Python scripts and require a ROCm-enabled GPU. `test_rocwmma_patch.py` checks correctness against PyTorch matmul and reports relative error; keep new kernels within the existing <1% error tolerance. Use `test_fragment_loading.py` when modifying fragment loading or layout logic. Run tests after rebuilding the extension to ensure you are exercising the latest kernels.

## Commit & Pull Request Guidelines
Use the repository's short `type: subject` convention, for example `docs: refresh profiling guidance` or `perf: reduce LDS stalls`. Include scope details in the body when needed. For pull requests, include the target GPU/ROCm version, numerical contract, matrix sizes tested, fresh-process distribution, and exact commands run; link related issues and retain raw results for promoted claims.

## Environment & Configuration Notes
The checked-in container reproduces the historical ROCm 7.9 pip environment and mixes it with ROCm 6.3 APT tooling. New performance work should use one consistent, gfx1151-capable stack (currently ROCm 7.14), record exact package/compiler versions, and compare against the historical toolchain when compiler effects matter. `hipcc` is supported; current ROCm also supports invoking AMD Clang directly. `build_and_test.sh` uses `ROCM_PATH` (defaults to `/opt/rocm`) and writes tuning artifacts to `/tmp/tunableop` via `TUNABLEOP_RESULTS_DIR`. Keep these paths in mind when running on shared systems.

For performance claims, follow `docs/PERFORMANCE_STATUS.md`: correctness first, unprofiled timing separate from counter collection, five fresh-process repeats, raw result retention, and compiled ISA/resource inspection. The 41.322 TFLOPS result is a validated peak, but it has not passed the strict sustained promotion gate. Do not present it as five-process sustained performance, and do not revive the unverified historical 21.9 TFLOPS mention as a record.
