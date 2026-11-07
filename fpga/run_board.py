"""`python3 -m nanoxl.driver` with the as-built FPGA configuration (fpga/boards.py) instead of the ADR-003
planning configs in nanoxl/hwconfig.py. Use this on the board: the compiler must not place data beyond the
bitstream's scratchpad capacities (e.g. the PYNQ-Z2 bitstream has a 128 KiB ABUF, the planning config 192 KiB).

    python3 fpga/run_board.py --board kv260 --backend pynq --bitstream nanoxl.bit \\
        --bundle artifacts/models/tiny/w8a8.npz --prompt "ROMEO:" --n 100
    python3 fpga/run_board.py --kv ddr ...          # DDR-KV mode (shakespeare on the KV260, anything on PYNQ-Z2)

All other options are the driver CLI's (docs/driver.md §1).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nanoxl.driver.__main__ as cli  # noqa: E402
from fpga.boards import RTL_PARAMS, hwconfig  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    kv = None
    if "--kv" in argv:
        i = argv.index("--kv")
        kv = {"onchip": True, "ddr": False}[argv[i + 1]]
        del argv[i:i + 2]
    cli.BOARDS = {b: hwconfig(b, kv) for b in RTL_PARAMS}   # the CLI looks its --board up here
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
