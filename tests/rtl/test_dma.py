"""DMA engine (rtl/dma_engine.sv): AXI4 master, realigner, 4 KiB splitting."""
import pytest

from simrun import run

DMA_SRC = ["rtl/fifo_sync.sv", "rtl/cmd_fields.sv", "rtl/dma_realign.sv", "rtl/dma_engine.sv"]


@pytest.mark.parametrize("maxb,ldf", [(16, 32), (256, 512)])
def test_dma_engine(maxb, ldf):
    run("dma_engine", DMA_SRC, "dma_cocotb", parameters={"MAXB": maxb, "LDF": ldf},
        extra_env={"DMA_MAXB": str(maxb), "DMA_LDF": str(ldf), "DMA_NCMD": "40"})
