#!/usr/bin/env python3
"""Pipeline completed VMEM refills into LDS after the overwrite barrier."""

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).read_text()
    old = (
        "\t;;#ASMSTART\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\t;;#ASMEND\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\ts_barrier\n"
        "\tds_store_b128 v68, v[74:77]\n"
        "\tds_store_b128 v68, v[78:81] offset:16\n"
        "\tds_store_b128 v73, v[114:117]\n"
    )
    new = (
        "\ts_barrier\n"
        "\ts_waitcnt vmcnt(2)\n"
        "\tds_store_b128 v68, v[74:77]\n"
        "\ts_waitcnt vmcnt(1)\n"
        "\tds_store_b128 v68, v[78:81] offset:16\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\tds_store_b128 v73, v[114:117]\n"
    )
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one hot-loop handoff, found {count}")
    output = source.replace(old, new, 1)
    Path(sys.argv[2]).write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
