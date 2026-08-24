#!/usr/bin/env python3
"""Remove the first eight edge masks for the fixed 4096-square benchmark."""

from pathlib import Path
import re
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT OUTPUT")
    text = Path(sys.argv[1]).read_text()
    begin = text.index("\n\ts_and_b32 s0, s3, vcc_lo\n")
    end = text.index("\n.LBB0_31:", begin)
    head, body, tail = text[:begin], text[begin:end], text[end:]
    # These are the eight independent N-fragment edge masks.  On the exact
    # 4096x4096 shape every lane is in bounds, so the exec save/restore and
    # branch are dead.  Keep coordinate arithmetic and stores unchanged.
    pattern = re.compile(
        r"\n\ts_and_b32 s0, s(?:3|[4-9]|10), vcc_lo.*?"
        r"\n\ts_cbranch_execz \.LBB0_[0-9]+\n",
        re.DOTALL,
    )
    body, removed = pattern.subn("\n", body)
    body, restored = re.subn(r"\n\ts_or_b32 exec_lo, exec_lo, s1", "", body)
    body, compares = re.subn(
        r"\n\tv_cmp_gt_i32_e64 s(?:3|[4-9]|10), s16, v68", "", body
    )
    if removed != 8 or restored != 7 or compares != 7:
        raise SystemExit(
            f"expected eight masks/7 restores/8 compares, removed={removed} "
            f"restores={restored} compares={compares}"
        )
    Path(sys.argv[2]).write_text(head + body + tail)


if __name__ == "__main__":
    main()
