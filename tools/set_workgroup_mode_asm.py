#!/usr/bin/env python3
"""Change a CU-mode code object to WGP mode without touching instructions."""

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).read_text()
    directives = (
        (
            "\t\t.amdhsa_workgroup_processor_mode 0\n",
            "\t\t.amdhsa_workgroup_processor_mode 1\n",
        ),
        (
            "    .workgroup_processor_mode: 0\n",
            "    .workgroup_processor_mode: 1\n",
        ),
    )
    for old, new in directives:
        if source.count(old) != 1:
            raise ValueError(f"expected one mode directive: {old.strip()}")
        source = source.replace(old, new, 1)

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
