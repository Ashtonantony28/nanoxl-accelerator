"""The nanoxl_top end-to-end smoke (tests/rtl/top_cocotb.py, unmodified) run through rtl/nanoxl_axi_wrap.v.

`WrapView` presents the wrapper under nanoxl_top's port names (clk, rst_n, s_axil_*, mf_axi_*, md_axi_*, md1_axi_*)
and forwards `u_vec` to the instance inside the wrapper, so the very same bench drives the Vivado-facing
port names. Passing it proves the shell is a pure rename: same driver, same golden comparisons
(logits, KV cache, per-op saturation counters, SAT CSRs). Environment: as top_cocotb (TOP_*).
"""
from __future__ import annotations

import cocotb

import top_cocotb

PREFIX = (("s_axil_", "s_axi_ctrl_"), ("mf_axi_", "m_axi_mf_"), ("md_axi_", "m_axi_md_"), ("md1_axi_", "m_axi_md1_"))
RENAME = {"clk": "aclk", "rst_n": "aresetn"}


def to_wrap(name: str) -> str:
    if name in RENAME:
        return RENAME[name]
    for old, new in PREFIX:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def to_top(name: str) -> str:
    for k, v in RENAME.items():
        if name == v:
            return k
    for old, new in PREFIX:
        if name.startswith(new):
            return old + name[len(new):]
    return name


class WrapView:
    """nanoxl_top-shaped view of a nanoxl_axi_wrap handle."""

    def __init__(self, dut):
        object.__setattr__(self, "_dut", dut)

    def __getattr__(self, name):
        d = object.__getattribute__(self, "_dut")
        if name == "u_vec":
            return d.u_top.u_vec
        if name.startswith("_"):
            return getattr(d, name)
        return getattr(d, to_wrap(name))

    def __dir__(self):
        d = object.__getattribute__(self, "_dut")
        return sorted({to_top(n) for n in dir(d)} | {"u_vec"})


@cocotb.test()
async def wrap_smoke(dut):
    assert dut._name == "nanoxl_axi_wrap"
    # the wrapper must not expose nanoxl_top's own port names (it is a rename, not a second interface)
    for n in ("clk", "rst_n", "s_axil_awaddr", "mf_axi_araddr", "md_axi_awaddr", "md1_axi_awaddr"):
        assert not hasattr(dut, n), n
    await top_cocotb.top_tiny.func(WrapView(dut))
