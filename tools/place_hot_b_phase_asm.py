#!/usr/bin/env python3
"""Place hot-loop B independently from four phase-4 A fragment banks."""

from pathlib import Path
import re
import sys


GROUPS = {
    "a0": (76, 83),
    "a1": (84, 91),
    "b": (92, 99),
    "a2": (100, 107),
    "a3": (108, 115),
    "refill": (116, 119),
}


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s B_BASE", file=sys.stderr)
        return 2

    b_base = int(sys.argv[3])
    if b_base not in (108, 111, 112):
        raise ValueError(
            "B_BASE must be 108 (phase 4), 111 (phase 7), or 112 (phase 0)"
        )

    source = Path(sys.argv[1]).read_text()
    start = source.index(".LBB0_13:")
    end = source.index("; %bb.14:", start)
    prefix = source[:start]
    hot = source[start:end]

    for anchor in (
        "\tds_load_b128 v[76:79], v68\n",
        "\tds_load_b128 v[92:95], v67 offset:12288\n",
        "\tbuffer_load_b128 v[116:119], v73, s[8:11], s18 offen\n",
    ):
        if hot.count(anchor) != 1:
            raise ValueError(f"selected-loop anchor {anchor.strip()!r} changed")

    targets = {
        "a0": 76,
        "a1": 84,
        "b": b_base,
        "a2": 92,
        "a3": 100,
        "refill": 72,
    }

    def map_register(register: int) -> int:
        for name, (first, last) in GROUPS.items():
            if first <= register <= last:
                return targets[name] + register - first
        if register == 73:
            return 65
        if register == 75:
            return 66
        return register

    def map_range(match: re.Match[str]) -> str:
        first = int(match.group(1))
        last = int(match.group(2))
        mapped = [map_register(register)
                  for register in range(first, last + 1)]
        if mapped != list(range(mapped[0], mapped[-1] + 1)):
            raise ValueError(f"range crosses placement boundary: {match.group(0)}")
        return f"v[{mapped[0]}:{mapped[-1]}]"

    hot = re.sub(r"\bv\[(\d+):(\d+)\]", map_range, hot)
    hot = re.sub(
        r"\bv(\d+)\b",
        lambda match: f"v{map_register(int(match.group(1)))}",
        hot,
    )
    shadows = "\tv_mov_b32_e32 v65, v73\n\tv_mov_b32_e32 v66, v75\n"
    source = prefix + shadows + hot + source[end:]

    registers = [int(value) for value in re.findall(r"\bv(\d+)\b", source)]
    registers.extend(
        int(value)
        for pair in re.findall(r"v\[(\d+):(\d+)\]", source)
        for value in pair
    )
    if max(registers) > 119:
        raise ValueError(f"placed image addresses v{max(registers)}")
    for needle, count in (
        ("\tv_wmma_f16_16x16x16_f16", 16),
        ("\tds_load_b128", 16),
        ("\tbuffer_load_b128", 3),
        ("\tds_store_b128", 3),
        ("\ts_barrier", 2),
    ):
        if hot.count(needle) != count:
            raise ValueError(f"hot loop must retain {count} {needle.strip()}")

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
