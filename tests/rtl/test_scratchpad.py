import pytest

from simrun import run


@pytest.mark.parametrize("width,depth,lat", [(32, 16, 1), (72, 64, 2), (256, 32, 1)])
def test_scratchpad(width, depth, lat):
    run("scratchpad", ["rtl/scratchpad.sv"], "scratchpad_cocotb",
        parameters={"WIDTH": width, "DEPTH": depth, "RD_LAT": lat},
        extra_env={"SP_TOP": "scratchpad", "SP_WIDTH": str(width), "SP_DEPTH": str(depth),
                   "SP_LAT": str(lat)})


@pytest.mark.parametrize("w,depth", [(16, 4), (33, 13), (40, 32)])
def test_scratchpad_fifo(w, depth):
    run("spad_fifo", ["rtl/scratchpad.sv"], "scratchpad_cocotb",
        parameters={"W": w, "DEPTH": depth},
        extra_env={"SP_TOP": "spad_fifo", "FIFO_W": str(w), "FIFO_DEPTH": str(depth)})
