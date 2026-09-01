#!/usr/bin/env python3
"""Create reproducible FP16-input/FP32-output Tensile schedule variants.

The source is AMD's Strix Halo HHS library logic.  Each ``--variant`` is a
JSON object of top-level solution overrides; ``ProblemType.<key>`` addresses a
nested problem-type field.  Derived-parameter flags are cleared so the pinned
Tensile revision recomputes layout and resource metadata after every override.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


def apply_overrides(solution: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if key.startswith(("AssertSize", "AssertStride")) and isinstance(value, dict):
            value = {int(index): item for index, item in value.items()}
        if key.startswith("ProblemType."):
            solution["ProblemType"][key.removeprefix("ProblemType.")] = value
        else:
            solution[key] = value


def adapt(
    source: Path,
    base_index: int,
    variants: list[dict],
    *,
    rederive: bool,
) -> list:
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

    try:
        base = next(s for s in data[5] if s["SolutionIndex"] == base_index)
    except StopIteration as exc:
        raise ValueError(f"solution index not present: {base_index}") from exc

    data[4]["DestDataType"] = 0
    problem_overrides = {
        key.removeprefix("ProblemType."): value
        for key, value in variants[0].items()
        if key.startswith("ProblemType.")
    }
    for overrides in variants[1:]:
        current = {
            key.removeprefix("ProblemType."): value
            for key, value in overrides.items()
            if key.startswith("ProblemType.")
        }
        if current != problem_overrides:
            raise ValueError(
                "all variants in one logic file need identical ProblemType overrides"
            )
    data[4].update(problem_overrides)
    solutions = []
    for index, overrides in enumerate(variants):
        solution = copy.deepcopy(base)
        solution["ProblemType"]["DestDataType"] = 0
        apply_overrides(solution, overrides)

        if rederive:
            # Force Tensile to rederive the LDS layout, work-group geometry,
            # and register metadata. Keeping AMD's cached values after changing
            # a schedule parameter can silently generate the wrong experiment.
            #
            # AMD's committed library logic stores one shared VectorWidth for
            # solution 91. Some schedule overrides make Tensile's rederivation
            # take a path that expects the per-tensor aliases to exist already
            # (KernelWriterAssembly.initKernel indexes them directly). Preserve
            # the committed meaning explicitly so a valid mapping/stagger
            # experiment is not mislabeled as a code-generation failure.
            if "VectorWidth" in solution:
                solution.setdefault("VectorWidthA", solution["VectorWidth"])
                solution.setdefault("VectorWidthB", solution["VectorWidth"])
            solution["AssignedDerivedParameters"] = False
            solution["AssignedProblemIndependentDerivedParameters"] = False
            solution["ProblemType"]["AssignedDerivedParameters"] = False
        solution["SolutionIndex"] = index
        # Library-logic inputs carry a cached minimum name.  Tensile expects
        # that field even while recomputing parameters, and variants must not
        # alias one another in the generated code-object manifest.
        base_name = solution.get("SolutionNameMin", f"solution_{base_index}")
        solution["SolutionNameMin"] = (
            base_name.replace("HHS", "HSS") + f"_VAR{index}"
        )
        if solution.get("SolutionName"):
            solution["SolutionName"] = (
                solution["SolutionName"].replace("HHS", "HSS")
                + f"_VAR{index}"
            )
        solutions.append(solution)

    data[5] = solutions
    contiguous_4096 = [4096, 4096, 1, 4096, 4096, 4096, 4096, 4096]
    data[7] = [[contiguous_4096, [0, 1.0]]]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-solution", type=int, default=34)
    parser.add_argument(
        "--keep-derived",
        action="store_true",
        help="retain AMD's cached derived layout for predicate-only variants",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="JSON object containing solution overrides; repeat for a sweep",
    )
    args = parser.parse_args()

    variants = [json.loads(item) for item in args.variant]
    if not all(isinstance(item, dict) for item in variants):
        raise ValueError("each --variant must decode to a JSON object")
    data = adapt(
        args.source,
        args.base_solution,
        variants,
        rederive=not args.keep_derived,
    )
    args.output.write_text(
        yaml.safe_dump(data, sort_keys=False, width=120), encoding="utf-8"
    )
    print(f"wrote {len(data[5])} HSS variant(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
