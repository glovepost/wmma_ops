#!/usr/bin/env python3
"""Share one 8-KiB K-tile offset between A and stride-padded B refills."""

from pathlib import Path
import sys


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).read_text()
    source = replace_once(
        source,
        "\ts_lshl_b64 s[8:9], s[8:9], 12\n",
        "\ts_lshl_b64 s[8:9], s[8:9], 13\n",
        "stride-padded B block base",
    )
    source = replace_once(
        source,
        "\ts_movk_i32 s7, 0x2000\n\ts_movk_i32 s18, 0x1000\n",
        "\ts_movk_i32 s7, 0x2000\n",
        "initial refill offsets",
    )
    source = replace_once(
        source,
        "\tbuffer_load_b128 v[116:119], v73, s[8:11], s18 offen\n",
        "\tbuffer_load_b128 v[116:119], v73, s[8:11], s7 offen\n",
        "B refill offset",
    )
    source = replace_once(
        source,
        "\ts_addk_i32 s7, 0x2000\n\ts_addk_i32 s18, 0x1000\n",
        "\ts_addk_i32 s7, 0x2000\n",
        "hot-loop refill increments",
    )

    if source.count("s18 offen") != 0:
        raise SystemExit("a B refill still uses the compact 4-KiB stride")
    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
