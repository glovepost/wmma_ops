#!/usr/bin/env python3
"""Diagnostic benchmark for persistent [N,K] prepacked weights.

This is intentionally not the row-major record contract: B is transposed once
before warm-up and timing, matching an inference engine with persistent packed
weights.  Results must always be labelled as prepacked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--candidate",
        choices=("interleaved", "k32"),
        default="interleaved",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import wmma_ops

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    a = torch.randn((args.m, args.k), dtype=torch.float16, device=device)
    b = torch.randn((args.k, args.n), dtype=torch.float16, device=device)
    reference = a.float() @ b.float()
    b_transposed = b.transpose(0, 1).contiguous()
    torch.cuda.synchronize()

    function_name = {
        "interleaved": "matmul_opt_interleaved_prepacked_bt",
        "k32": "matmul_opt_k32_prepacked_bt",
    }[args.candidate]
    function = getattr(wmma_ops, function_name)
    result = function(a, b_transposed)
    torch.cuda.synchronize()
    difference = result - reference
    max_abs = difference.abs().max().item()
    reference_max = reference.abs().max().item()
    correctness = {
        "finite": bool(torch.isfinite(result).all().item()),
        "max_abs_error": max_abs,
        "reference_max_abs": reference_max,
        "normalized_max_error": max_abs / reference_max,
        "rms_error": difference.square().mean().sqrt().item(),
    }
    correctness["pass"] = (
        correctness["finite"] and correctness["normalized_max_error"] < 0.01
    )
    if not correctness["pass"]:
        raise RuntimeError(f"correctness rejected: {correctness}")

    for _ in range(args.warmup):
        _ = function(a, b_transposed)
    torch.cuda.synchronize()

    operations = 2.0 * args.m * args.n * args.k
    samples = []
    for _ in range(args.blocks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            result = function(a, b_transposed)
        end.record()
        end.synchronize()
        time_ms = start.elapsed_time(end) / args.iterations
        samples.append({
            "time_ms": time_ms,
            "tflops": operations / (time_ms * 1.0e-3) / 1.0e12,
        })

    tflops = [sample["tflops"] for sample in samples]
    report = {
        "schema_version": 1,
        "benchmark_class": "persistent-prepacked-weight diagnostic",
        "record_comparable": False,
        "prepack_outside_timing": True,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "input_dtype": "float16",
        "accumulator_output_dtype": "float32",
        "candidate": function_name,
        "correctness": correctness,
        "samples": samples,
        "median_tflops": statistics.median(tflops),
        "min_tflops": min(tflops),
        "max_tflops": max(tflops),
        "cv_percent": (
            100.0 * statistics.stdev(tflops) / statistics.fmean(tflops)
            if len(tflops) > 1 else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"prepacked diagnostic: {report['median_tflops']:.3f} TFLOPS "
        f"({report['min_tflops']:.3f}-{report['max_tflops']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
