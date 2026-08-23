#!/usr/bin/env python3
"""Reproducible gfx1151 WMMA record benchmark.

The historical record used a 4096-cubed FP16-input, FP32-output GEMM. This
script keeps that contract, validates every candidate against an FP32 PyTorch
reference, and saves block-level timing samples for repeated fresh-process runs.
Profiler or counter collection must be performed separately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from typing import Any, Callable


DEFAULT_CANDIDATES = ("matmul", "matmul_adaptive", "matmul_opt", "matmul_zerocopy")
CAPTURED_ENVIRONMENT = (
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ARCH",
    "ROCM_PATH",
    "WMMA_CU_MODE",
    "WMMA_OPT_MIN_BLOCKS_PER_CU",
    "WMMA_UNROLL_THRESHOLD",
)
ROCM_SMI_COMMAND = (
    "rocm-smi",
    "--showproductname",
    "--showdriverversion",
    "--showclocks",
    "--showpower",
    "--showtemp",
)


def command_output(command: list[str], timeout: int = 15) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout + result.stderr).strip()
    return output or None


def git_metadata() -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": mean,
        "p75": percentile(values, 0.75),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "cv_percent": (
            100.0 * statistics.stdev(values) / mean
            if len(values) > 1 and mean
            else 0.0
        ),
    }


def time_block(
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    result: torch.Tensor | None = None
    start.record()
    for _ in range(iterations):
        result = function(a, b)
    end.record()
    end.synchronize()
    assert result is not None
    return start.elapsed_time(end) / iterations, result


def correctness_metrics(result: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    difference = result.float() - reference
    max_abs = difference.abs().max().item()
    reference_max = reference.abs().max().item()
    normalized_max = max_abs / reference_max if reference_max else math.inf
    rms = difference.square().mean().sqrt().item()
    finite = bool(torch.isfinite(result).all().item())
    return {
        "finite": finite,
        "max_abs_error": max_abs,
        "reference_max_abs": reference_max,
        "normalized_max_error": normalized_max,
        "rms_error": rms,
        "pass": finite and normalized_max < 0.01,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100,
                        help="Iterations in each timing block")
    parser.add_argument("--blocks", type=int, default=5,
                        help="Timing blocks per candidate")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--run-id", default="0")
    parser.add_argument("--candidate", action="append", dest="candidates",
                        help="wmma_ops function to test; repeat for multiple candidates")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.warmup, args.iterations, args.blocks) <= 0:
        parser.error("dimensions, warmup, iterations, and blocks must be positive")
    return args


def main() -> int:
    global torch
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        print(f"failed to import PyTorch: {error}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ROCm device is not available through PyTorch", file=sys.stderr)
        return 2

    try:
        import wmma_ops
    except ImportError as error:
        print(f"failed to import wmma_ops: {error}", file=sys.stderr)
        return 2

    candidate_names = tuple(args.candidates or DEFAULT_CANDIDATES)
    missing = [name for name in candidate_names if not hasattr(wmma_ops, name)]
    if missing:
        print(f"wmma_ops is missing candidates: {', '.join(missing)}", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)

    a = torch.randn((args.m, args.k), dtype=torch.float16, device=device)
    b = torch.randn((args.k, args.n), dtype=torch.float16, device=device)
    reference = torch.matmul(a.float(), b.float())
    torch.cuda.synchronize()
    before_smi = command_output(list(ROCM_SMI_COMMAND))

    functions = {name: getattr(wmma_ops, name) for name in candidate_names}
    correctness: dict[str, dict[str, Any]] = {}
    for name, function in functions.items():
        result = function(a, b)
        torch.cuda.synchronize()
        correctness[name] = correctness_metrics(result, reference)
        del result

    accepted = [name for name in candidate_names if correctness[name]["pass"]]
    rejected = [name for name in candidate_names if not correctness[name]["pass"]]
    if rejected:
        print(f"correctness rejected: {', '.join(rejected)}", file=sys.stderr)

    for name in accepted:
        function = functions[name]
        for _ in range(args.warmup):
            _ = function(a, b)
        torch.cuda.synchronize()

    timing_jobs = [name for name in accepted for _ in range(args.blocks)]
    random.Random(f"{args.seed}:{args.run_id}").shuffle(timing_jobs)
    samples: dict[str, list[dict[str, float]]] = {name: [] for name in accepted}
    operations = 2.0 * args.m * args.n * args.k
    for name in timing_jobs:
        time_ms, result = time_block(functions[name], a, b, args.iterations)
        tflops = operations / (time_ms * 1.0e-3) / 1.0e12
        samples[name].append({"time_ms": time_ms, "tflops": tflops})
        del result

    after_smi = command_output(list(ROCM_SMI_COMMAND))

    candidates: dict[str, Any] = {}
    for name in candidate_names:
        tflops_values = [sample["tflops"] for sample in samples.get(name, [])]
        time_values = [sample["time_ms"] for sample in samples.get(name, [])]
        candidates[name] = {
            "correctness": correctness[name],
            "samples": samples.get(name, []),
            "tflops": summarize(tflops_values) if tflops_values else None,
            "time_ms": summarize(time_values) if time_values else None,
        }

    report = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "benchmark_contract": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "input_dtype": "float16",
            "accumulator_output_dtype": "float32",
            "reference": "torch.matmul(A.float(), B.float())",
            "metric": "HIP event elapsed time around repeated Python binding calls",
            "fresh_output_per_call": True,
            "warmup_per_candidate": args.warmup,
            "iterations_per_block": args.iterations,
            "blocks_per_candidate": args.blocks,
            "seed": args.seed,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "device_name": properties.name,
            "device_total_memory": properties.total_memory,
            "environment": {
                key: os.environ[key]
                for key in CAPTURED_ENVIRONMENT
                if key in os.environ
            },
            "git": git_metadata(),
            "hipcc_version": command_output(["hipcc", "--version"]),
            "rocm_smi_before": before_smi,
            "rocm_smi_after": after_smi,
        },
        "candidate_order": timing_jobs,
        "candidates": candidates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for name in accepted:
        summary = candidates[name]["tflops"]
        print(
            f"{name}: median {summary['median']:.3f} TFLOPS "
            f"(range {summary['min']:.3f}-{summary['max']:.3f}, "
            f"CV {summary['cv_percent']:.2f}%)"
        )
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
