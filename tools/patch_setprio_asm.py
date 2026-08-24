#!/usr/bin/env python3
"""Add the gfx11 MAC-cluster priority window to a hand-scheduled WMMA loop."""

from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT OUTPUT")
    text = Path(sys.argv[1]).read_text()
    start_loop = "\n.LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"
    start_final = "\n.LBB0_15:\n"
    if text.count(start_loop) != 1 or text.count(start_final) != 1:
        raise SystemExit("unexpected loop labels")
    text = text.replace(start_loop, start_loop + "\ts_setprio 1\n", 1)
    text = text.replace(start_final, start_final + "\ts_setprio 1\n", 1)
    # The first barrier is the producer/consumer handoff.  Restore priority
    # immediately before it in both the steady-state and final-tile paths.
    steady = "\n\ts_barrier\n\ts_waitcnt vmcnt(2)\n"
    if text.count(steady) != 1:
        raise SystemExit("steady-state handoff not found")
    text = text.replace(steady, "\n\ts_setprio 0\n\ts_barrier\n\ts_waitcnt vmcnt(2)\n", 1)
    final = "\n\ts_and_b32 s0, s3, vcc_lo\n"
    if text.count(final) != 1:
        raise SystemExit("final compute boundary not found")
    text = text.replace(final, "\n\ts_setprio 0\n\ts_and_b32 s0, s3, vcc_lo\n", 1)
    Path(sys.argv[2]).write_text(text)


if __name__ == "__main__":
    main()
