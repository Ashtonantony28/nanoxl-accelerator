"""M1 sizing study: run the cycle model over array sizes, DDR bandwidths, boards, models, weight
widths, KV placement and decode batch; write CSV/JSON tables and a roofline PNG to bench/results/.

    python3 bench/roofline.py            # full sweep (~1-2 min on 4 cores)
    python3 bench/roofline.py --quick    # board table only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoxl.cyclemodel import estimate  # noqa: E402
from nanoxl.graph import SHAKESPEARE, TINY  # noqa: E402
from nanoxl.hwconfig import BOARDS, KV260, PYNQ_Z2  # noqa: E402

OUT = ROOT / "bench" / "results"
MODELS = {"tiny": TINY, "shakespeare": SHAKESPEARE}
COLS = ["table", "board", "model", "wbits", "N", "lanes", "mhz", "ddr_gbps", "kv", "batch", "fits",
        "prefill_T", "prefill_chunk", "prefill_ms", "prefill_array_util", "prefill_gops",
        "decode_tok_s", "weight_bound_tok_s", "decode_step_us", "decode_ddr_util", "decode_ddr_MB_per_step",
        "decode_ops_per_byte", "decode_gops", "why"]


def job(args):
    table, board, model, wbits, kv, batch, over, prefill = args
    hw = BOARDS[board].with_(kv_onchip=(kv == "onchip"), **over)
    cfg = MODELS[model].with_(wbits=wbits)
    r = estimate(cfg, hw, batch=batch, prefill_len=None if prefill else 0, n_pos=3)
    r["table"] = table
    return r


def jobs(quick):
    out = []
    # A: each board as proposed, every model / width / KV placement / batch
    for board in BOARDS:
        for model in MODELS:
            for wbits in (8, 4):
                for kv in ("onchip", "ddr"):
                    for batch in (1, 4, 8):
                        out.append(("boards", board, model, wbits, kv, batch, {}, batch == 1))
    if quick:
        return out
    # B: sizing sweep, array N x DDR bandwidth, at each board's clock / lanes
    for board in BOARDS:
        for n in (8, 16, 32, 64):
            for bw in (1.5, 4, 8, 12):
                for wbits in (8, 4):
                    kv = "onchip" if board == "kv260" else "ddr"
                    out.append(("sweep", board, "shakespeare", wbits, kv, 1, {"array_n": n, "ddr_gbps": bw}, True))
    # C: weight-load port width (rows per cycle) at KV260 for the bandwidth sweep
    for wl in (1, 2):
        for bw in (4, 8, 12):
            out.append(("wload", "kv260", "shakespeare", 8, "onchip", 1,
                        {"wload_rows_per_cycle": wl, "ddr_gbps": bw}, False))
    return out


def fmt(v):
    return round(v, 4) if isinstance(v, float) else v


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; CSV only")
        return False
    ink, muted, grid = "#2b2b2a", "#6b6a64", "#e4e3dd"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # left: roofline for KV260 (N=32, 250 MHz) and PYNQ-Z2 (N=16, 150 MHz)
    ax = axes[0]
    import numpy as np
    oi = np.logspace(-0.5, 3, 200)
    for k, (hw, c) in enumerate(((KV260, series[0]), (PYNQ_Z2, series[1]))):
        ax.plot(oi, np.minimum(hw.peak_gops, hw.ddr_gbps * oi), color=c, lw=2, label=f"{hw.name} roof")
        pts = [r for r in rows if r["table"] == "boards" and r["board"] == hw.name and r["fits"]
               and r["model"] == "shakespeare"
               and r["kv"] == ("onchip" if hw.kv_onchip and r["batch"] == 1 else "ddr")]
        for r in pts:
            mk = "o" if r["wbits"] == 8 else "s"
            ax.scatter(r["decode_ops_per_byte"], r["decode_gops"], s=46, marker=mk, color=c,
                       edgecolor="white", linewidth=1.5, zorder=3)
            if r["wbits"] == 8:
                ax.annotate(f"B={r['batch']}", (r["decode_ops_per_byte"], r["decode_gops"]),
                            textcoords="offset points", xytext=(6, -10), fontsize=8, color=muted)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operational intensity (ops / DDR byte)", color=ink)
    ax.set_ylabel("GOPS", color=ink)
    ax.set_title("Decode roofline, shakespeare, B = 1/4/8 (circle INT8, square INT4)", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    # right: decode tok/s vs DDR bandwidth per array N (KV260 clock/lanes, INT8)
    ax = axes[1]
    for k, n in enumerate((8, 16, 32, 64)):
        pts = sorted((r["ddr_gbps"], r["decode_tok_s"]) for r in rows if r["table"] == "sweep"
                     and r["board"] == "kv260" and r["N"] == n and r["wbits"] == 8)
        ax.plot(*zip(*pts), color=series[k], lw=2, marker="o", ms=6, label=f"N={n}")
    bws = [1.5, 4, 8, 12]
    ax.plot(bws, [b * 1e9 / SHAKESPEARE.weight_bytes for b in bws], color=muted, lw=1.5, ls="--",
            label="weight-stream bound")
    ax.set_xlabel("DDR bandwidth (GB/s)", color=ink)
    ax.set_ylabel("decode tokens/s (batch 1)", color=ink)
    ax.set_title("KV260 clock/lanes, shakespeare INT8: array size vs bandwidth", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        a.grid(True, color=grid, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(muted)
        a.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    js = jobs(a.quick)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, js))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "roofline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in COLS})
    (OUT / "roofline.json").write_text(json.dumps([{k: fmt(v) for k, v in r.items()} for r in rows], indent=1))
    if not a.quick:
        plot(rows, OUT / "roofline.png")
    # board summary table (markdown) for the ADR
    lines = ["| board | model | W | KV | B | prefill ms (T) | prefill util | decode tok/s | weight bound | DDR util |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["table"] != "boards":
            continue
        if not r["fits"]:
            lines.append(f"| {r['board']} | {r['model']} | {r['wbits']} | {r['kv']} | {r['batch']} | KV does not fit | | | | |")
            continue
        pf = f"{r['prefill_ms']:.2f} ({r['prefill_T']}, chunk {r['prefill_chunk']})" if "prefill_ms" in r else ""
        pu = f"{r['prefill_array_util']:.2f}" if "prefill_ms" in r else ""
        lines.append(f"| {r['board']} | {r['model']} | {r['wbits']} | {r['kv']} | {r['batch']} | {pf} | {pu} | "
                     f"{r['decode_tok_s']:.0f} | {r['weight_bound_tok_s']:.0f} | {r['decode_ddr_util']:.2f} |")
    (OUT / "boards.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
