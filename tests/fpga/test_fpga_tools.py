"""Board-side tools of the FPGA flow, exercised against the CSR emulator (SimBackend): the first-light
checklist (fpga/first_light.py) and the driver CLI wrapper with the as-built configs (fpga/run_board.py)."""
import json

import pytest

from fpga import first_light, run_board


@pytest.mark.parametrize("board", ["kv260", "pynq-z2"])
def test_first_light_sim(board, capsys):
    assert first_light.main(["--board", board, "--backend", "sim", "--n", "6"]) == 0
    out = capsys.readouterr().out
    for i, step in enumerate(("csr", "end", "dma", "bandwidth", "layer", "generate"), 1):
        assert f"[{i}] {step:9s} PASS" in out, out


def test_first_light_reports_failure(monkeypatch, capsys):
    """A broken loopback must stop the checklist with FAIL (not pass silently)."""
    real = first_light.step_dma

    def broken(be, dev, nbytes=16384):
        off, _ = first_light._scratch(dev)
        orig = be.invalidate

        def corrupt(*a, **k):
            orig(*a, **k)
            dev.mem[off + first_light.MiB + 5] ^= 0xFF
        be.invalidate = corrupt
        return real(be, dev, nbytes)
    monkeypatch.setattr(first_light, "step_dma", broken)
    assert first_light.main(["--board", "kv260", "--backend", "sim"]) == 1
    out = capsys.readouterr().out
    assert "[3] dma       FAIL" in out and "[4]" not in out


@pytest.mark.parametrize("board,kv", [("pynq-z2", "onchip"), ("kv260", "ddr")])
def test_run_board_cli(board, kv, capsys):
    ref = json.loads(open(first_light.TINY / "golden_ref.json").read())
    toks = ",".join(map(str, ref["prompt"]))
    assert run_board.main(["--board", board, "--kv", kv, "--bundle", str(first_light.TINY / "w8a8.npz"),
                           "--tokens", toks, "--n", "4", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["tokens"] == ref["bundles"]["w8a8"]["tokens"][:4]
