"""Shared cocotb helpers for the vector-unit benches: lane packing and a cycle-driven
valid/ready stream bench with random valid gaps and random backpressure.

Timing discipline: inputs are driven after the falling edge; handshakes are sampled in
ReadOnly of the same half cycle, i.e. exactly what the next rising edge will see.
Sources obey the valid/ready rule: once valid is raised, the beat is held until accepted.
While valid is low, sources drive random junk on their data pins so that sampling
data without valid is caught.
"""
from __future__ import annotations
import os
import random

from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


def pack(vals, w):
    r = 0
    m = (1 << w) - 1
    for i, v in enumerate(vals):
        r |= (int(v) & m) << (i * w)
    return r


def unpack(x, n, w, signed=True):
    m = (1 << w) - 1
    out = []
    for i in range(n):
        v = (x >> (i * w)) & m
        if signed and v >> (w - 1):
            v -= 1 << w
        out.append(v)
    return out


def u(sig):
    v = sig.value
    return v.to_unsigned() if hasattr(v, "to_unsigned") else int(v)


def env_int(name, default):
    return int(os.environ.get(name, default))


async def start(dut, zero=()):
    """Start a 10 ns clock, zero the named inputs, apply a synchronous reset."""
    from cocotb import start_soon
    start_soon(Clock(dut.clk, 10, unit="ns").start())
    for n in zero:
        getattr(dut, n).value = 0
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1


class Source:
    """beats: list of items; drive(item) sets data pins; junk() randomizes them."""

    def __init__(self, valid, ready, beats, drive, junk=None, p=0.75, rng=None):
        self.valid, self.ready, self.beats = valid, ready, list(beats)
        self.drive, self.junk, self.p = drive, junk, p
        self.rng = rng or random.Random(0)
        self.i = 0
        self.holding = False

    def done(self):
        return self.i >= len(self.beats)

    def step_drive(self):
        if not self.holding and not self.done() and self.rng.random() < self.p:
            self.holding = True
            self.drive(self.beats[self.i])
        if not self.holding and self.junk is not None:
            self.junk()
        self.valid.value = 1 if self.holding else 0

    def step_sample(self):
        if self.holding and self.ready.value == 1:
            self.i += 1
            self.holding = False


class Sink:
    """capture() reads the data pins when a beat transfers; results go to .got."""

    def __init__(self, valid, ready, capture, p=0.75, rng=None):
        self.valid, self.ready, self.capture, self.p = valid, ready, capture, p
        self.rng = rng or random.Random(1)
        self.got = []
        self.rdy = 0

    def step_drive(self):
        self.rdy = 1 if self.rng.random() < self.p else 0
        self.ready.value = self.rdy

    def step_sample(self):
        if self.rdy and self.valid.value == 1:
            self.got.append(self.capture())


async def run_bench(dut, sources, sinks, expect_counts, max_cycles=2_000_000, hooks=()):
    """Run until every sink has collected its expected number of beats.

    hooks: objects with step_drive()/step_sample() (e.g. a behavioural SFU model)."""
    cycles = 0
    while any(len(s.got) < n for s, n in zip(sinks, expect_counts)):
        await FallingEdge(dut.clk)
        for o in (*sources, *sinks, *hooks):
            o.step_drive()
        await ReadOnly()
        for o in (*sources, *sinks, *hooks):
            o.step_sample()
        cycles += 1
        assert cycles < max_cycles, "bench timeout (deadlock?)"
    # A few idle cycles: nothing extra may come out.
    for s in sinks:
        s.p = 1.0
    for _ in range(12):
        await FallingEdge(dut.clk)
        for o in sources:
            o.valid.value = 0
        for o in (*sinks, *hooks):
            o.step_drive()
        await ReadOnly()
        for o in (*sinks, *hooks):
            o.step_sample()
    for s, n in zip(sinks, expect_counts):
        assert len(s.got) == n, f"sink produced {len(s.got)} beats, expected {n}"
    await FallingEdge(dut.clk)
    return cycles
