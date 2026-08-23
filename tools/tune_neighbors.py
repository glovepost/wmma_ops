#!/usr/bin/env python3
"""Screen the exact one-parameter neighborhood of the FP16 gfx1151 leader."""

import argparse
import math
import sys

sys.path.insert(0, "../rocm_wmma_gemm/config")

from tune import WMMATuner  # noqa: E402


BASELINE = (4, 2, 4, 4, 1, 0, 16, 128)
TFLOPS_MS_NUMERATOR = 137.438953472


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=1)
    args = parser.parse_args()

    tuner = WMMATuner(65536, "gfx1151", 42, "f16_f16")
    valid = [
        config
        for config in tuner.valid_configs
        if tuner._check_constraints(config, 1, 0, 0)
    ]
    configs = sorted(
        config
        for config in valid
        if sum(a != b for a, b in zip(config, BASELINE)) == args.distance
    )
    print(f"candidate_count={len(configs)}", flush=True)
    for index, config in enumerate(configs, 1):
        milliseconds = tuner._evaluate_config(
            4096, 4096, 4096, 1, 0, 0, config
        )
        tflops = (
            TFLOPS_MS_NUMERATOR / milliseconds
            if math.isfinite(milliseconds)
            else 0.0
        )
        print(
            f"RESULT {index:02d} config={config} "
            f"ms={milliseconds:.6f} tflops={tflops:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
