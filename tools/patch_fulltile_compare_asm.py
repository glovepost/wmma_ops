#!/usr/bin/env python3
"""Remove dead edge compares while preserving the generated exec masks."""

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
    body, count = re.subn(
        r"\n\tv_cmp_gt_i32_e64 s(?:4|[5-9]|10), s16, v68", "", body
    )
    if count != 7:
        raise SystemExit(f"expected seven late edge compares, removed={count}")
    Path(sys.argv[2]).write_text(head + body + tail)


if __name__ == "__main__":
    main()
