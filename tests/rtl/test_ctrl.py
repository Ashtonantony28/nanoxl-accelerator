"""Control plane top (rtl/ctrl_top.sv): CSR + fetch + queues/semaphores + DMA, engine stubs."""
import os

import pytest

from simrun import run

CTRL_SRC = ["rtl/fifo_sync.sv", "rtl/cmd_fields.sv", "rtl/cmd_decode.sv", "rtl/cmd_ring_ptr.sv",
            "rtl/cmd_fetch.sv", "rtl/sem_unit.sv", "rtl/cmd_queue.sv", "rtl/cmd_proc.sv",
            "rtl/dma_realign.sv", "rtl/dma_engine.sv", "rtl/csr_axil.sv", "rtl/ctrl_top.sv"]


def test_ctrl_top():
    run("ctrl_top", CTRL_SRC, "ctrl_cocotb", parameters={"MAXB": 16, "LDF": 32},
        extra_env={"CTRL_NCMD": "300"})


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("SIM", "icarus") != "verilator", reason="long run; Verilator only")
def test_ctrl_top_long():
    run("ctrl_top", CTRL_SRC, "ctrl_cocotb", extra_env={"CTRL_NCMD": "3000"}, tag="_long")
