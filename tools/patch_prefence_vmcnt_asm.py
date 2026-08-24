#!/usr/bin/env python3
"""Move the first refill VMEM wait ahead of the publish barrier."""

from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT OUTPUT")
    text = Path(sys.argv[1]).read_text()
    old = "\n\ts_barrier\n\ts_waitcnt vmcnt(2)\n\tds_store_b128 v70, v[76:79]\n"
    new = "\n\ts_waitcnt vmcnt(2)\n\ts_barrier\n\tds_store_b128 v70, v[76:79]\n"
    if text.count(old) != 1:
        raise SystemExit("publish barrier/wait sequence not unique")
    Path(sys.argv[2]).write_text(text.replace(old, new, 1))


if __name__ == "__main__":
    main()
