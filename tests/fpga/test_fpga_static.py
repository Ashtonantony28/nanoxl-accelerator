"""Static checks of the FPGA flow (SG9a, docs/fpga.md): no Vivado here, so these check what can be checked.

* the Vivado build scripts run end to end in tclsh against logging stubs of every Vivado command
  (syntax, argument parsing, procs), and pass exactly fpga/boards.py's parameters, part, clock and
  CSR base;
* the RTL list of the scripts exists and equals the set nanoxl_axi_wrap elaborates from (Yosys hierarchy);
* the wrapper's ports are nanoxl_top's ports, renamed to AXI names Vivado infers, with matching
  X_INTERFACE_INFO attributes, widths and directions; it lints clean in Verilator;
* the as-built HwConfig fits the RTL capacities and the compiled tiny model runs (funcsim) == golden;
* bench/results/resources_*.json were produced for the current board parameters.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fpga import boards
from fpga.yosys.prep import main as prep_main

ROOT = Path(__file__).resolve().parents[2]
WRAP = ROOT / "rtl" / "nanoxl_axi_wrap.v"
BUILD = {"kv260": ROOT / "fpga" / "kv260" / "build.tcl", "pynq-z2": ROOT / "fpga" / "pynqz2" / "build.tcl"}
STUB = Path(__file__).parent / "vivado_stub.tcl"
needs_tclsh = pytest.mark.skipif(shutil.which("tclsh") is None, reason="tclsh not installed (apt install tcl)")
needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not installed")


def tcl_rtl_list() -> list[str]:
    txt = (ROOT / "fpga" / "common" / "rtl_files.tcl").read_text()
    body = re.search(r"set NANOXL_RTL \{(.*?)\}", txt, re.S).group(1)
    return body.split()


# ---------------------------------------------------------------- build scripts (tclsh dry run)
def dry_run(board, tmp_path, *extra):
    log = tmp_path / f"{board}.log"
    r = subprocess.run(["tclsh", str(STUB), str(log), str(BUILD[board]), "-out", str(tmp_path / "proj"),
                        "-jobs", "2", *extra], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = log.read_text().splitlines()
    assert lines[-1] == "STUB_DONE"
    return lines


def cell_config(lines, cell):
    for ln in lines:
        if ln.startswith("set_property -dict") and ln.endswith(f" {cell}"):
            return dict(re.findall(r"CONFIG\.(\w+) \{?([^ }]+)\}?", ln))
    raise AssertionError(f"no set_property -dict for {cell}")


@needs_tclsh
@pytest.mark.parametrize("board", sorted(BUILD))
def test_build_tcl_dry_run(board, tmp_path):
    lines = dry_run(board, tmp_path)
    log = "\n".join(lines)
    # nanoxl_0 parameters == fpga/boards.py
    cfg = cell_config(lines, "nanoxl_0")
    assert {k: int(v) for k, v in cfg.items()} == boards.RTL_PARAMS[board]
    assert "create_bd_cell -type module -reference nanoxl_axi_wrap nanoxl_0" in log
    assert f"-part {boards.PART[board]}" in log
    # clock, CSR window, HP ports, interrupt
    if board == "kv260":
        ps = cell_config(lines, "zynq_ultra_ps_e_0")
        assert float(ps["PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ"]) == boards.CLOCK_MHZ[board]
        assert ps["PSU__SAXIGP2__DATA_WIDTH"] == ps["PSU__SAXIGP3__DATA_WIDTH"] == "128"
        assert ps["PSU__SAXIGP0__DATA_WIDTH"] == "128"
        hp = {"m_axi_mf": ("smc_mf", "S_AXI_HPC0_FPD"), "m_axi_md": ("smc_md", "S_AXI_HP0_FPD"),
              "m_axi_md1": ("smc_md1", "S_AXI_HP1_FPD")}
        assert "zynq_ultra_ps_e_0/pl_ps_irq0" in log
    else:
        ps = cell_config(lines, "processing_system7_0")
        assert float(ps["PCW_FPGA0_PERIPHERAL_FREQMHZ"]) == boards.CLOCK_MHZ[board]
        hp = {"m_axi_mf": ("smc_hp0", "S_AXI_HP0"), "m_axi_md": ("smc_hp0", "S_AXI_HP0"),
              "m_axi_md1": ("smc_md1", "S_AXI_HP1")}
        assert "processing_system7_0/IRQ_F2P" in log
    assert re.search(rf"-offset {boards.CSR_BASE[board]:#010x} -range 4K nanoxl_0/s_axi_ctrl/\*", log, re.I), \
        "CSR window"
    for intf, (smc, port) in hp.items():   # ADR-005 §4 board mapping, through a SmartConnect each
        assert re.search(rf"connect_bd_intf_net \S*nanoxl_0/{intf}\S* \S*{smc}/S0[01]_AXI", log), intf
        assert re.search(rf"connect_bd_intf_net \S*{smc}/M00_AXI\S* \S*/{port}\b", log), intf
        assert re.search(rf"assign_bd_address -target_address_space nanoxl_0/{intf} \S*{port}/\S*DDR", log), intf
    for pin in ("aclk", "aresetn", "irq"):
        assert f"nanoxl_0/{pin}" in log
    # every nanoxl_0 pin the script touches is a wrapper port / interface
    ports = {p["name"] for p in wrap_ports()}
    intfs = {"s_axi_ctrl", "m_axi_mf", "m_axi_md", "m_axi_md1"}
    for name in re.findall(r"nanoxl_0/(\w+)", log):
        assert name in ports or name in intfs, name
    # the flow reaches the bitstream and exports the overlay pair under one base name
    assert "launch_runs impl_1 -to_step write_bitstream" in log
    assert re.search(r"file copy -force \S+\.bit \S+/overlay/nanoxl\.bit", log)
    assert re.search(r"file copy -force \S+\.hwh \S+/overlay/nanoxl\.hwh", log)
    assert "report_timing_summary" in log
    # the RTL the project reads is the checked-in list
    added = re.search(r"add_files -norecurse -fileset sources_1 \{([^}]*)\}", log).group(1).split()
    assert [str(Path(a).relative_to(ROOT)) for a in added] == tcl_rtl_list()


@needs_tclsh
def test_build_tcl_options(tmp_path):
    """-clk / -params overrides reach the PS and the core; -proj_only stops before synthesis; bad args fail."""
    lines = dry_run("kv260", tmp_path, "-clk", "200", "-params", "N 16 L 16", "-proj_only")
    assert cell_config(lines, "zynq_ultra_ps_e_0")["PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ"] == "200"
    cfg = cell_config(lines, "nanoxl_0")
    assert cfg["N"] == "16" and cfg["L"] == "16" and cfg["KV_LINES"] == "65536"
    assert not any(ln.startswith("launch_runs") for ln in lines)
    r = subprocess.run(["tclsh", str(STUB), str(tmp_path / "x.log"), str(BUILD["kv260"]), "-bogus"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "unknown argument" in r.stderr


# ---------------------------------------------------------------- RTL file set
def module_files() -> dict[str, str]:
    out = {}
    for f in sorted((ROOT / "rtl").glob("*.sv")) + [WRAP]:
        for m in re.findall(r"^\s*module\s+(\w+)", f.read_text(), re.M):
            out[m] = f"rtl/{f.name}"
    return out


@needs_yosys
@pytest.mark.parametrize("board", sorted(BUILD))
def test_rtl_fileset_matches_hierarchy(board, tmp_path):
    lst = tcl_rtl_list()
    for f in lst:
        assert (ROOT / f).is_file(), f
    assert len(set(lst)) == len(lst)
    srcs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "rtl").glob("*.sv"))
    prep_main(["prep", str(tmp_path / "rtl"), *[str(ROOT / s) for s in srcs]])
    files = " ".join(str(tmp_path / "rtl" / Path(s).name) for s in srcs)
    chp = " ".join(f"-set {k} {v}" for k, v in boards.RTL_PARAMS[board].items())
    ls = tmp_path / "ls.txt"
    script = (f"read_verilog -sv -DSYNTHESIS {files}\nread_verilog {WRAP}\n"
              f"chparam {chp} nanoxl_axi_wrap\nhierarchy -top nanoxl_axi_wrap\ntee -q -o {ls} ls\n")
    (tmp_path / "h.ys").write_text(script)
    r = subprocess.run(["yosys", "-q", "-s", str(tmp_path / "h.ys")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    mods = set()
    for ln in ls.read_text().splitlines()[1:]:
        ln = ln.strip()
        if ln and not ln.endswith("modules:"):
            mods.add(re.sub(r"^\$paramod(\$[0-9a-f]+)?\\", "", ln).split("\\")[0])
    mf = module_files()
    need = {mf[m] for m in mods}
    assert need == set(lst), f"missing {sorted(need - set(lst))}, extra {sorted(set(lst) - need)}"
    # and the simulations use the same set
    from test_top import TOP_SRC
    assert set(TOP_SRC) | {"rtl/nanoxl_axi_wrap.v"} == set(lst)


# ---------------------------------------------------------------- wrapper ports
def _ports(text, kind):
    """[(dir, width expr, name, attrs)] of a module header."""
    if kind == "v":
        rx = r"((?:\(\*.*?\*\)\s*)*)(input|output)\s+wire\s*(\[[^\]]+\])?\s*(\w+)"
    else:
        rx = r"()(input|output)\s+logic\s*(\[[^\]]+\])?\s*(\w+)"
    hdr = text[text.index(") ("):text.index(");", text.index(") ("))]
    return [{"attrs": a, "dir": d, "width": (w or "").replace(" ", ""), "name": n}
            for a, d, w, n in re.findall(rx, hdr, re.S)]


def wrap_ports():
    return _ports(WRAP.read_text(), "v")


AXIL = {"AWADDR", "AWPROT", "AWVALID", "AWREADY", "WDATA", "WSTRB", "WVALID", "WREADY", "BRESP", "BVALID",
        "BREADY", "ARADDR", "ARPROT", "ARVALID", "ARREADY", "RDATA", "RRESP", "RVALID", "RREADY"}
AXI_R = {"ARID", "ARADDR", "ARLEN", "ARSIZE", "ARBURST", "ARLOCK", "ARCACHE", "ARPROT", "ARVALID", "ARREADY",
         "RID", "RDATA", "RRESP", "RLAST", "RVALID", "RREADY"}
AXI_W = {"AWID", "AWADDR", "AWLEN", "AWSIZE", "AWBURST", "AWLOCK", "AWCACHE", "AWPROT", "AWVALID", "AWREADY",
         "WDATA", "WSTRB", "WLAST", "WVALID", "WREADY", "BID", "BRESP", "BVALID", "BREADY"}
# signals a master drives (a slave receives them)
MASTER_OUT = {s for s in AXIL | AXI_R | AXI_W if not s.endswith("READY") and s[0] in "AW"} | \
             {"RREADY", "BREADY"}


def test_wrapper_ports_follow_vivado_axi_naming():
    top = _ports((ROOT / "rtl" / "nanoxl_top.sv").read_text(), "sv")
    wp = wrap_ports()
    assert len(wp) == len(top)
    from wrap_cocotb import to_wrap
    by = {p["name"]: p for p in wp}
    for t in top:   # same port, renamed, same direction and width
        w = by[to_wrap(t["name"])]
        assert (w["dir"], w["width"]) == (t["dir"], t["width"]), t["name"]
    want = {"s_axi_ctrl": (AXIL, "slave"), "m_axi_mf": (AXI_R, "master"), "m_axi_md": (AXI_R | AXI_W, "master"),
            "m_axi_md1": (AXI_R | AXI_W, "master")}
    seen = {k: set() for k in want}
    params = {}
    for p in wp:
        n, a = p["name"], p["attrs"]
        if n in ("aclk", "aresetn", "irq"):
            kind = {"aclk": "clock", "aresetn": "reset", "irq": "interrupt"}[n]
            assert f'"xilinx.com:signal:{kind}:1.0 {n} ' in a, n
            continue
        m = re.fullmatch(r"(s_axi_ctrl|m_axi_mf|m_axi_md|m_axi_md1)_([a-z]+)", n)
        assert m, f"{n}: not <interface>_<signal>"
        intf, sig = m.group(1), m.group(2).upper()
        assert f'X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 {intf} {sig}"' in a, n
        pm = re.search(r'X_INTERFACE_PARAMETER = "([^"]*)"', a)
        if pm:
            params[intf] = pm.group(1)
        seen[intf].add(sig)
        role = want[intf][1]
        drives = sig in MASTER_OUT
        assert p["dir"] == ("output" if drives == (role == "master") else "input"), n
    for intf, (sigs, _) in want.items():
        assert seen[intf] == sigs, (intf, sigs ^ seen[intf])
    clk = next(p for p in wp if p["name"] == "aclk")["attrs"]
    assert "ASSOCIATED_BUSIF s_axi_ctrl:m_axi_mf:m_axi_md:m_axi_md1," in clk and "ASSOCIATED_RESET aresetn" in clk
    assert "POLARITY ACTIVE_LOW" in next(p for p in wp if p["name"] == "aresetn")["attrs"]
    assert "PROTOCOL AXI4LITE" in params["s_axi_ctrl"] and "ADDR_WIDTH 12" in params["s_axi_ctrl"]
    assert "READ_ONLY" in params["m_axi_mf"] and "DATA_WIDTH 128" in params["m_axi_md"]
    assert "READ_WRITE" in params["m_axi_md1"]
    # the checked-in wrapper is the generator's output for today's nanoxl_top
    from fpga.gen_wrap import render
    assert WRAP.read_text() == render(), "rtl/nanoxl_axi_wrap.v is stale: run python3 fpga/gen_wrap.py"
    # Verilog-2001: no SystemVerilog keywords in the shell (module references need a Verilog top)
    body = re.sub(r"\(\*.*?\*\)", "", re.sub(r"//.*", "", WRAP.read_text()), flags=re.S)
    for kw in ("logic", "always_ff", "always_comb", "int ", "typedef", "interface "):
        assert kw not in body, kw
    # parameters: the same names and defaults as nanoxl_top
    tp = dict(re.findall(r"parameter int (\w+)\s*=\s*(\d+)", (ROOT / "rtl/nanoxl_top.sv").read_text()))
    wpp = dict(re.findall(r"parameter integer (\w+)\s*=\s*(\d+)", WRAP.read_text()))
    assert tp == wpp


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_wrapper_lint():
    from test_top import TOP_SRC
    r = subprocess.run(["verilator", "--lint-only", "-Wall", "-Wno-fatal", "-Wno-DECLFILENAME", "-Wno-UNUSEDPARAM",
                        "--top-module", "nanoxl_axi_wrap", str(WRAP), *[str(ROOT / s) for s in TOP_SRC]],
                       capture_output=True, text=True, cwd=ROOT)
    msgs = [ln for ln in r.stderr.splitlines() if "nanoxl_axi_wrap.v" in ln]
    assert not msgs, "\n".join(msgs)
    assert r.returncode == 0 and "%Error" not in r.stderr, r.stderr[-3000:]   # 42 known warnings below (docs/top.md §7)


# ---------------------------------------------------------------- as-built HwConfig
@pytest.mark.parametrize("board", sorted(BUILD))
def test_hwconfig_fits_rtl_and_runs_tiny(board):
    from nanoxl.driver import SimBackend, open_bundle
    ref = json.loads((ROOT / "artifacts/models/tiny/golden_ref.json").read_text())
    cap = boards.capacities(board)
    p = boards.RTL_PARAMS[board]
    for kv in (True, False):
        hw = boards.hwconfig(board, kv)
        assert hw.array_n == p["N"] and hw.lanes == p["L"] and hw.wload_rows_per_cycle == 1
        for k, v in cap.items():
            assert getattr(hw, k) <= v, k
            assert v & (v - 1) == 0, f"{k} capacity must be a power of two"
        for k in ("ABUF_LINES", "WBUF_LINES", "ACC_LINES", "PBUF_LINES", "KV_LINES"):
            assert p[k] <= 1 << 17, "scratchpad line addresses are 17 bits"
        dev = open_bundle(str(ROOT / "artifacts/models/tiny/w8a8.npz"), SimBackend(hw), hw)
        assert dev.generate(ref["prompt"], 3) == ref["bundles"]["w8a8"]["tokens"][:3]


@pytest.mark.parametrize("board", sorted(BUILD))
def test_resource_estimates_current(board):
    """bench/results/resources_<board>.json exists and was made with today's board parameters
    (regenerate: python3 fpga/yosys/estimate.py <board>)."""
    f = ROOT / "bench" / "results" / f"resources_{board}.json"
    assert f.is_file(), f"run python3 fpga/yosys/estimate.py {board}"
    res = json.loads(f.read_text())
    assert res["params"] == boards.RTL_PARAMS[board]
    assert set(res["blocks"]) >= {"ctrl_top", "array_seq", "vec_seq", "spad_router", "glue"}
    assert not res.get("failed"), res.get("failed")
    for k in ("LUT", "FF", "DSP", "BRAM36", "URAM"):
        assert k in res["utilization"]


@pytest.mark.parametrize("board", sorted(BUILD))
def test_estimate_mirrors_top_ports(board):
    """fpga/yosys/estimate.py's buf_spad port widths (RPL / WPL) are nanoxl_top's localparams."""
    from fpga.yosys.estimate import blocks
    top = (ROOT / "rtl" / "nanoxl_top.sv").read_text()
    p = boards.RTL_PARAMS[board]
    N = p["N"]
    val = {"1": 1, "VP": p["VP"], "CPL": N // 4 if N >= 4 else 1, "WPL": (N + 15) // 16, "APL": (N + 15) // 16}

    def vec(name):
        body = re.search(rf"localparam logic \[\w+\*8-1:0\] {name}\s*=\s*\{{([^}}]*)\}}", top).group(1)
        toks = [re.sub(r"^8'(?:d(\d+)|\((\w+)\))$", lambda m: m.group(1) or m.group(2), t.strip())
                for t in body.split(",")]
        v = [val[t] for t in reversed(toks)]           # Verilog concat: last element is port 0
        return f"{8 * len(v)}'h" + "".join(f"{x:02x}" for x in reversed(v)), len(v)

    router = next(b for b in blocks(p) if b[0] == "spad_router")[2]
    (rpl, nr), (wpl, nw) = vec("RPL"), vec("WPLV")
    assert (router["RPL"], router["NR"], router["WPL"], router["NW"]) == (rpl, nr, wpl, nw)
