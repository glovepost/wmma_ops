#!/usr/bin/env python3
"""Offset the one-dimensional workgroup ID at a selected kernel's entry."""

import argparse
from pathlib import Path


ENTRY = "; %bb.0:\n\ts_load_b128 s[16:19], s[0:1], 0x18\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("offset", type=lambda value: int(value, 0))
    args = parser.parse_args()

    if not 1 <= args.offset <= 0x7FFF:
        raise ValueError("S_ADDK_I32 offset must be in [1, 32767]")

    source = args.input.read_text()
    if source.count(ENTRY) != 1:
        raise ValueError("expected one fixed kernel entry sequence")
    if "; One-time workgroup-ID offset." in source:
        raise ValueError("input already contains a workgroup-ID offset")

    replacement = (
        "; %bb.0:\n"
        "; One-time workgroup-ID offset. The mapper consumes s2 immediately\n"
        "; below; no later instruction observes the unshifted system value.\n"
        f"\ts_addk_i32 s2, 0x{args.offset:x}\n"
        "\ts_load_b128 s[16:19], s[0:1], 0x18\n"
    )
    args.output.write_text(source.replace(ENTRY, replacement, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
