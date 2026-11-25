#!/usr/bin/env python3
"""README media for NanoXL.

  python3 scripts/media.py [--out docs/media]

  generate.{gif,mp4}   the compiled tiny model generating text through the host driver's SimBackend
                       (funcsim of the 128-bit command stream, bit-identical to the golden model and to
                       the Verilator RTL per tests/); device time and DMA bytes are the driver's model
  dataflow.{gif,mp4}   the weight-stationary dataflow of rtl/systolic_array.sv (N = 8): a Python model of
                       pe.sv + skew/deskew, checked against numpy, drawn cycle by cycle
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BG, PANEL, GRID, TXT, DIM = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
ACC, A_COL, W_COL, P_COL = "#3fb950", "#58a6ff", "#f0883e", "#d2a8ff"


def generate(out, prompt="ROMEO:", n=160, fps=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nanoxl.driver.__main__ import BOARDS, CharCodec, SimBackend, open_bundle
    hw = BOARDS["kv260-fpga"]                         # as-built first bitstream (ADR-006)
    dev = open_bundle(str(ROOT / "artifacts" / "models" / "tiny" / "w8a8.npz"), SimBackend(hw), hw)
    codec = CharCodec()
    toks = dev.generate(codec.encode(prompt), n)
    perf = dev.perf()
    text = codec.decode(toks)
    us_per_tok = perf["cycles"] / hw.clock_mhz / len(toks)
    tmp = Path(tempfile.mkdtemp())
    per = 2                                           # frames per token
    nfr = len(text) * per + fps * 3
    for f in range(nfr):
        k = min(len(text), f // per)
        fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
        fig.text(0.04, 0.92, "NanoXL: a trained nanoGPT on the INT8 accelerator", fontsize=18, weight="bold", color=TXT)
        fig.text(0.04, 0.875, "FUNCTIONAL SIMULATION of the compiled command stream (bit-exact vs the integer golden model and the "
                 "Verilator RTL)  ·  greedy decoding  ·  device time is modelled", fontsize=9.5, color=DIM)
        ax = fig.add_axes([0.04, 0.08, 0.60, 0.74])
        ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(GRID)
        body = prompt + text[:k]
        lines, cur = [], ""
        for ch in body:
            if ch == "\n" or len(cur) >= 58:
                lines.append(cur)
                cur = "" if ch == "\n" else ch
            else:
                cur += ch
        lines.append(cur + ("▌" if (f // 6) % 2 == 0 or k < len(text) else " "))
        lines = lines[-14:]
        for i, ln in enumerate(lines):
            if i == 0 and ln.startswith(prompt) and len(lines) < 14:
                ax.text(0.03, 0.93 - i * 0.065, prompt, color=ACC, fontsize=13, family="DejaVu Sans Mono", transform=ax.transAxes,
                        va="top", weight="bold")
                ax.text(0.03 + len(prompt) * 0.0137, 0.93 - i * 0.065, ln[len(prompt):], color=TXT, fontsize=13,
                        family="DejaVu Sans Mono", transform=ax.transAxes, va="top")
            else:
                ax.text(0.03, 0.93 - i * 0.065, ln, color=TXT, fontsize=13, family="DejaVu Sans Mono", transform=ax.transAxes,
                        va="top")
        stats = [("model", "tiny: 4 layers, d = 128"), ("quantisation", "W8A8 (INT8 weights and activations)"),
                 ("target", f"KV260 first bitstream: N = {hw.array_n}, {hw.clock_mhz:.0f} MHz"), ("tokens", f"{k} / {len(text)}"),
                 ("command programs", f"{max(1, k)}"),
                 ("modelled device time", f"{us_per_tok * k / 1e3:6.2f} ms ({1e6 / us_per_tok:,.0f} tok/s)"),
                 ("DMA traffic", f"{perf['bytes'] / len(text) * k / 1e6:5.1f} MB")]
        for i, (a, b) in enumerate(stats):
            fig.text(0.68, 0.76 - i * 0.085, a, color=DIM, fontsize=10)
            fig.text(0.68, 0.725 - i * 0.085, b, color=TXT, fontsize=12.5, family="DejaVu Sans Mono")
        fig.savefig(tmp / f"f{f:04d}.png", facecolor=BG)
        plt.close(fig)
    mp4, gif = out / "generate.mp4", out / "generate.gif"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%04d.png"), "-c:v", "libx264",
                    "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=10,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=32:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=none:diff_mode=rectangle", str(gif)], check=True)


def ws_array(A, Wt, N):
    """Cycle model of the weight-stationary array: PE(r, c) holds W[r, c]; activation row m enters row r
    skewed by r cycles and moves right; partial sums move down; column c leaves deskewed. Returns the
    per-cycle state (a, psum) and the output C = A @ W."""
    M = A.shape[0]
    a = np.zeros((N, N), int); a_v = np.zeros((N, N), bool); p = np.zeros((N, N), int)
    states, out = [], np.zeros((M, N), int)
    T = M + 2 * N + 1
    for t in range(T):
        na = np.zeros_like(a); nv = np.zeros_like(a_v); npsum = np.zeros_like(p)
        for r in range(N):
            for c in range(N):
                if c == 0:
                    m = t - r
                    ain, vin = (A[m, r], True) if 0 <= m < M else (0, False)
                else:
                    ain, vin = a[r, c - 1], a_v[r, c - 1]
                pin = p[r - 1, c] if r > 0 else 0
                na[r, c], nv[r, c] = ain, vin
                npsum[r, c] = pin + (ain * Wt[r, c] if vin else 0)
        a, a_v, p = na, nv, npsum
        for c in range(N):                                     # bottom row output: row m of C at t = m + (N-1) + c
            m = t - (N - 1) - c
            if 0 <= m < M:
                out[m, c] = p[N - 1, c]
        states.append((a.copy(), a_v.copy(), p.copy()))
    return states, out


def dataflow(out, N=8, M=8, fps=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    A = rng.integers(-128, 128, (M, N)); W = rng.integers(-128, 128, (N, N))
    states, C = ws_array(A, W, N)
    assert np.array_equal(C, A @ W), "array model disagrees with numpy"
    tmp = Path(tempfile.mkdtemp())
    vmax = np.abs(A @ W).max()
    frames = list(range(len(states))) + [len(states) - 1] * (fps * 2)
    for f, t in enumerate(frames):
        a, av, p = states[t]
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
        ax = fig.add_axes([0.20, 0.06, 0.50, 0.78])
        ax.set_facecolor(BG); ax.set_xlim(-2.6, N + 0.4); ax.set_ylim(-N - 2.4, 1.4); ax.set_aspect("equal"); ax.axis("off")
        for r in range(N):
            for c in range(N):
                act = av[r, c]
                ax.add_patch(plt.Rectangle((c + 0.06, -r - 0.94), 0.88, 0.88, color=PANEL, ec=W_COL if act else GRID, lw=1.6 if act else 0.6))
                if act:
                    ax.add_patch(plt.Rectangle((c + 0.06, -r - 0.94), 0.88, 0.88 * min(1, abs(p[r, c]) / vmax), color=P_COL, alpha=0.35))
                    ax.text(c + 0.5, -r - 0.32, f"{a[r, c]:d}", color=A_COL, fontsize=8, ha="center", family="DejaVu Sans Mono")
                ax.text(c + 0.5, -r - 0.72, f"w{W[r, c]:d}", color=W_COL, fontsize=6.5, ha="center", alpha=0.8, family="DejaVu Sans Mono")
            m = t - r
            if 0 <= m < M:
                ax.annotate("", xy=(0.02, -r - 0.5), xytext=(-0.9, -r - 0.5), arrowprops=dict(arrowstyle="->", color=A_COL, lw=1.5))
                ax.text(-1.0, -r - 0.5, f"A[{m},{r}]", color=A_COL, fontsize=8, ha="right", va="center", family="DejaVu Sans Mono")
        done = sum(1 for m in range(M) for c in range(N) if t >= m + N - 1 + c)
        for c in range(N):
            m = t - (N - 1) - c
            if 0 <= m < M:
                ax.annotate("", xy=(c + 0.5, -N - 0.9), xytext=(c + 0.5, -N - 0.06), arrowprops=dict(arrowstyle="->", color=P_COL, lw=1.5))
                ax.text(c + 0.5, -N - 1.3, f"C[{m},{c}]", color=P_COL, fontsize=7, ha="center", family="DejaVu Sans Mono")
        fig.text(0.03, 0.93, "NanoXL systolic array: weight-stationary dataflow", fontsize=18, weight="bold", color=TXT)
        fig.text(0.03, 0.895, f"N = {N} (the RTL default), INT8 × INT8 → INT32. Python cycle model of rtl/pe.sv + skew/deskew, "
                 "output checked against numpy A @ W", fontsize=9.5, color=DIM)
        side = [("cycle", f"{t:2d}"), ("MACs this cycle", f"{int(av.sum()):2d} / {N * N}"), ("outputs done", f"{done:2d} / {M * N}")]
        for i, (k_, v) in enumerate(side):
            fig.text(0.74, 0.72 - i * 0.10, k_, color=DIM, fontsize=11)
            fig.text(0.74, 0.68 - i * 0.10, v, color=TXT, fontsize=16, family="DejaVu Sans Mono")
        for i, (lab, col) in enumerate((("activations move right (row r skewed by r)", A_COL),
                                        ("weights stay in each PE", W_COL), ("partial sums move down", P_COL))):
            fig.text(0.74, 0.34 - i * 0.05, "■ " + lab, color=col, fontsize=10)
        fig.savefig(tmp / f"f{f:03d}.png", facecolor=BG)
        plt.close(fig)
    mp4, gif = out / "dataflow.mp4", out / "dataflow.gif"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%03d.png"), "-c:v", "libx264",
                    "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%03d.png"), "-vf",
                    "scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none", str(gif)],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dataflow(out)
    generate(out)
    for f in sorted(out.iterdir()):
        print(f, f"{f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
