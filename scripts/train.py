#!/usr/bin/env python3
"""Train an FP32 nanoGPT (Shakespeare-char) on CPU.

Default nanoGPT schedule (linear warmup + cosine decay to min_lr), AdamW(0.9, 0.99),
wd 0.1 on 2-D params only, grad clip 1.0. Time-boxable with --max-minutes: the cosine
schedule is defined over --iters, and training stops early if the wall-clock box is hit
(the best-val checkpoint is kept).

  python3 scripts/train.py --model tiny --out artifacts/models/tiny/ckpt.pt
  python3 scripts/train.py --model shakespeare --iters 2500 --batch 16 --max-minutes 75 \
      --out artifacts/ckpt/shakespeare/ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanoxl.model_fp import CONFIGS, GPT, load_shakespeare, save_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONFIGS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-every", type=int, default=0, help="also write <out>_last.pt every N iters")
    ap.add_argument("--resume", default=None, help="checkpoint to resume weights (+ optimizer if saved) from")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val, chars = load_shakespeare()
    cfg = CONFIGS[args.model]
    model = GPT(cfg)
    T = cfg.block_size
    start = 0
    ck = None
    if args.resume:
        from nanoxl.model_fp import load_nanogpt_state_dict
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_nanogpt_state_dict(model, ck["model"])
        start = int(ck["iter"]) + 1
        rng = np.random.default_rng(args.seed + start)   # fresh batch stream after resume
        print(f"resumed from {args.resume} at iter {start}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99))
    opt_path = Path(args.out).with_suffix(".opt.pt")
    if ck is not None and Path(args.resume).with_suffix(".opt.pt").exists():
        opt.load_state_dict(torch.load(Path(args.resume).with_suffix(".opt.pt"), weights_only=False))
        print("optimizer state restored", flush=True)

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / (args.warmup + 1)
        r = min(1.0, (it - args.warmup) / max(1, args.iters - args.warmup))
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)

    def batch(data, bs):
        ix = rng.integers(0, len(data) - T - 1, bs)
        x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
        return x, y

    eval_rng_state = np.random.default_rng(12345)
    vi = eval_rng_state.integers(0, len(val) - T - 1, args.eval_batches * args.batch)

    @torch.no_grad()
    def est_val():
        model.eval()
        losses = []
        for j in range(args.eval_batches):
            ix = vi[j * args.batch:(j + 1) * args.batch]
            x = torch.from_numpy(np.stack([val[i:i + T] for i in ix]).astype(np.int64))
            y = torch.from_numpy(np.stack([val[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
            losses.append(model(x, y)[1].item())
        model.train()
        return float(np.mean(losses))

    t0 = time.time()
    best = (float("inf"), -1)
    log = list(ck.get("log", [])) if ck is not None else []
    if ck is not None:
        best = (float(ck["val_est"]), start - 1)
    model.train()
    it = 0
    for it in range(start, args.iters + 1):
        stop = (time.time() - t0) / 60 > args.max_minutes
        if it % args.eval_every == 0 or it == args.iters or stop:
            vl = est_val()
            log.append({"iter": it, "val": vl, "min": (time.time() - t0) / 60})
            print(f"iter {it} val {vl:.4f} t {(time.time()-t0)/60:.1f}m", flush=True)
            if vl < best[0]:
                best = (vl, it)
                save_checkpoint(args.out, model, {"iter": it, "val_est": vl, "args": vars(args),
                                                  "chars": chars, "log": log})
                torch.save(opt.state_dict(), opt_path)
        if stop or it == args.iters:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        if args.save_every and it > start and it % args.save_every == 0:
            lp = Path(args.out).with_name(Path(args.out).stem + "_last.pt")
            save_checkpoint(lp, model, {"iter": it - 1, "val_est": best[0], "args": vars(args),
                                        "chars": chars, "log": log})
            torch.save(opt.state_dict(), lp.with_suffix(".opt.pt"))
        x, y = batch(train, args.batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            print(f"  it {it} loss {loss.item():.4f} lr {lr_at(it):.2e}", flush=True)
    summary = {"model": args.model, "best_val_est": best[0], "best_iter": best[1],
               "last_iter": it, "minutes": (time.time() - t0) / 60, "args": vars(args)}
    Path(args.out).with_suffix(".json").write_text(json.dumps({**summary, "log": log}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
