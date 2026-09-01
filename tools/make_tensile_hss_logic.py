#!/usr/bin/env python3
"""Adapt selected Strix Halo Tensile HHS solutions to FP32 output.

The input is AMD's ``strixhalo_Cijk_Ailk_Bljk_HHS_BH.yaml`` logic file. The
generated logic retains the FP16-input/FP32-accumulate schedule and changes only
the C/D destination type from half (HHS) to float (HSS). It is intended for
controlled experiments with the matching TensileCreateLibrary revision.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def parse_indices(specification: str) -> list[int]:
    """Parse comma-separated indices and half-open ranges such as 90:100."""
    result: list[int] = []
    for component in specification.split(","):
        component = component.strip()
        if not component:
            continue
        if ":" in component:
            start_text, stop_text = component.split(":", 1)
            result.extend(range(int(start_text), int(stop_text)))
        else:
            result.append(int(component))
    if not result or len(result) != len(set(result)):
        raise ValueError("solution selection must be non-empty and unique")
    return result


def adapt(source: Path, selected_indices: list[int], mapped_index: int) -> list:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 11:
        raise ValueError("input is not a supported Tensile library-logic file")

    problem = data[4]
    if (
        problem.get("DataType") != 4
        or problem.get("DestDataType") != 4
        or problem.get("ComputeDataType") != 0
        or not problem.get("HighPrecisionAccumulate")
    ):
        raise ValueError("expected AMD's FP16/FP16/FP16 HPA problem type")

    by_index = {solution["SolutionIndex"]: solution for solution in data[5]}
    missing = sorted(set(selected_indices) - set(by_index))
    if missing:
        raise ValueError(f"solution indices not present in input: {missing}")
    if mapped_index not in selected_indices:
        raise ValueError("mapped solution must be included in --solutions")

    data[4]["DestDataType"] = 0
    adapted_solutions = []
    old_to_new: dict[int, int] = {}
    for new_index, old_index in enumerate(selected_indices):
        solution = copy.deepcopy(by_index[old_index])
        solution["ProblemType"]["DestDataType"] = 0
        solution["SolutionIndex"] = new_index
        for key in ("SolutionName", "SolutionNameMin"):
            if key in solution:
                solution[key] = solution[key].replace("HHS", "HSS")
        adapted_solutions.append(solution)
        old_to_new[old_index] = new_index
    data[5] = adapted_solutions

    # Tensile's speed value is selection metadata. Preserve AMD's measured
    # value for the mapped 4096^3 solution while replacing the tuned padded
    # leading dimensions with this repository's contiguous record shape.
    mapped_speed = 1.0
    for sizes, selection in data[7]:
        if sizes[:4] == [4096, 4096, 1, 4096] and selection[0] == mapped_index:
            mapped_speed = float(selection[1])
            break
    contiguous_4096 = [4096, 4096, 1, 4096, 4096, 4096, 4096, 4096]
    data[7] = [[contiguous_4096, [old_to_new[mapped_index], mapped_speed]]]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="AMD Strix Halo HHS logic YAML")
    parser.add_argument("output", type=Path, help="path for adapted HSS YAML")
    parser.add_argument(
        "--solutions",
        default="90:100",
        help="comma-separated indices/half-open ranges (default: 90:100)",
    )
    parser.add_argument(
        "--mapped-solution",
        type=int,
        default=99,
        help="source solution selected for contiguous 4096^3 (default: 99)",
    )
    args = parser.parse_args()

    data = adapt(args.source, parse_indices(args.solutions), args.mapped_solution)
    args.output.write_text(
        yaml.safe_dump(data, sort_keys=False, width=120), encoding="utf-8"
    )
    print(f"wrote {len(data[5])} HSS solution(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
