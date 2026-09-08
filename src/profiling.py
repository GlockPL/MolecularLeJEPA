"""
Lightweight step profiler for the pretraining loop.

Enable with `profile: true` in the config. When enabled, accumulates wall
time per named section (with CUDA sync for accurate GPU timing) and the
training loop prints a breakdown every `log_every` steps.

When disabled, `section()` and `add()` are near-zero overhead, so the
instrumentation can stay in the hot path permanently.

Note: the CUDA syncs inserted around each section serialize CPU/GPU
overlap, so absolute totals run slightly slower than an un-profiled step.
The *relative* breakdown (share %) is the trustworthy signal.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class Profiler:
    def __init__(self) -> None:
        self.enabled = False
        self._times: dict[str, float] = defaultdict(float)
        self._steps = 0

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._times[name] += time.perf_counter() - t0

    def add(self, name: str, seconds: float) -> None:
        """Record a pre-measured duration (e.g. DataLoader wait time)."""
        if self.enabled:
            self._times[name] += seconds

    def end_step(self) -> None:
        if self.enabled:
            self._steps += 1

    def report(self) -> str:
        if not self._steps:
            return "(no profiling data)"
        total = sum(self._times.values())
        lines = [f"  {'section':16s} {'ms/step':>10s} {'share':>8s}"]
        for name, t in sorted(self._times.items(), key=lambda kv: -kv[1]):
            ms = t / self._steps * 1000.0
            share = (t / total * 100.0) if total else 0.0
            lines.append(f"  {name:16s} {ms:10.1f} {share:7.1f}%")
        lines.append(f"  {'TOTAL':16s} {total / self._steps * 1000.0:10.1f} {100.0:7.1f}%")
        return "\n".join(lines)

    def reset(self) -> None:
        self._times.clear()
        self._steps = 0


# Global singleton shared across the training loop, loss, and model.
PROFILER = Profiler()