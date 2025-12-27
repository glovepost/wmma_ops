# Repository Guidelines

## Project Structure & Module Organization
Core HIP/C++ kernels live in `wmma_gemm.hip` and the supporting headers `wmma_kernels_optimized.hpp`, `wmma_tile_selection.hpp`, and `wmma_device_helpers.hpp`. The custom rocWMMA header patch is in `rocwmma_patch/rocwmma_gfx1151.hpp`. Python entry points and benchmarking utilities are top-level scripts (for example, `autotune.py`, `benchmark_summary.py`, `rocprof_wmma.py`). Long-form notes live in `docs/WMMA_DEVELOPMENT_NOTES.md`, while external reference projects are under `examples/`.

## Build, Test, and Development Commands
- `python3 -m pip install -e . --no-build-isolation` — build and install the extension in editable mode.
- `./build_and_test.sh` — end-to-end build + import check + test run (expects ROCm + hipcc).
- `python3 test_rocwmma_patch.py` — main correctness/performance test suite.
- `python3 test_fragment_loading.py` — focused fragment-loading correctness checks.
- `python3 autotune.py --quick` — quick tuning run without Optuna.
- `python3 rocprof_wmma.py` — profiling run with rocprof helpers.

## Coding Style & Naming Conventions
Use 4-space indentation for both HIP/C++ and Python. Follow existing naming patterns: snake_case for functions/variables (for example, `wmma_gemm_kernel`), `wmma_*.hpp` for headers, and `test_*.py` for test scripts. There is no enforced formatter; keep changes consistent with adjacent code blocks and avoid style-only diffs.

## Testing Guidelines
Tests run as standalone Python scripts and require a ROCm-enabled GPU. `test_rocwmma_patch.py` checks correctness against PyTorch matmul and reports relative error; keep new kernels within the existing <1% error tolerance. Use `test_fragment_loading.py` when modifying fragment loading or layout logic. Run tests after rebuilding the extension to ensure you are exercising the latest kernels.

## Commit & Pull Request Guidelines
Git history currently has a single bootstrap commit, so there is no established commit message convention. Use a short, imperative summary and include scope details in the body if needed. For pull requests, include the target GPU/ROCm version, matrix sizes tested, and benchmark deltas when changing kernels; link related issues and list the exact commands run.

## Environment & Configuration Notes
ROCm 7.9/7.10-preview and `hipcc` on `PATH` are required. `build_and_test.sh` uses `ROCM_PATH` (defaults to `/opt/rocm`) and writes tuning artifacts to `/tmp/tunableop` via `TUNABLEOP_RESULTS_DIR`. Keep these paths in mind when running on shared systems.
