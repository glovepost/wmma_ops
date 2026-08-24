#!/usr/bin/env python3
"""Use a 64-bit recurrence for the first eight fixed-stride C stores."""

from pathlib import Path
import re
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT OUTPUT")
    text = Path(sys.argv[1]).read_text()
    marker = "\n\ts_lshl_b32 s14, s17, 1\n"
    if text.count(marker) != 1:
        raise SystemExit("epilogue stride marker not unique")
    text = text.replace(marker, marker + "\ts_lshl_b32 s19, s14, 1\n", 1)
    first = "\n\tglobal_store_b16 v[68:69], v57, off\n"
    if text.count(first) != 1:
        raise SystemExit("first epilogue store not unique")
    text = text.replace(
        first,
        "\n\tv_mov_b32_e32 v65, v68\n\tv_mov_b32_e32 v66, v69\n" + first,
        1,
    )
    # Edge checks remain intact.  Replace only the address reconstruction after
    # each check; v65:v66 carries the prior store pointer.
    for reg, label in zip(("v58", "v59", "v60", "v61", "v62", "v63", "v64"),
                          ("18", "20", "22", "24", "26", "28", "30")):
        pattern = re.compile(
            rf"(; %bb\.{label}:\n).*?"
            rf"\n\tglobal_store_b16 v\[68:69\], {reg}, off",
            re.DOTALL,
        )
        replacement = (
            rf"\1\tv_add_co_u32 v65, s0, v65, s19\n"
            "\tv_add_co_ci_u32_e64 v66, null, 0, v66, s0\n"
            "\tv_mov_b32_e32 v68, v65\n"
            "\tv_mov_b32_e32 v69, v66\n"
            rf"\tglobal_store_b16 v[68:69], {reg}, off"
        )
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise SystemExit(f"store {reg} address block not found")
    Path(sys.argv[2]).write_text(text)


if __name__ == "__main__":
    main()
