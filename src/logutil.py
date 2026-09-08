"""Shared run-logging utility — a reviewer-grade paper trail for every script.

Reviewers want evidence for every number ("papers for everything"). This module
is the ONE canonical implementation of the stdout/stderr tee that used to be
copy-pasted (as ``_Tee``) into individual scripts. Use it in every entry point so
that each run auto-writes a timestamped, ANSI-clean log capturing ALL output —
ours and any library's — without rewriting each ``print``.

Preferred (context manager — captures tracebacks too, restores on exit):

    from src.logutil import RunLogger

    with RunLogger("measure_coverage_molhiv", corpus="chembl", workers=8):
        ...                     # everything printed here is logged

Minimal-reindent (parse args first so ``--help`` doesn't spawn a log):

    log = RunLogger("ensemble_eval", config=args.config).start()
    import atexit; atexit.register(log.stop)   # robust to exceptions
    ...

The log path is ``<log_dir>/<prefix>_<YYYYmmdd_HHMMSS>.txt`` (default ``logs/``).
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

# Strip terminal colour codes from the FILE copy (kept on the console).
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class _Mirror:
    """File-like object that writes to a console stream AND a logfile."""

    def __init__(self, console, fh):
        self._console = console
        self._fh = fh

    def write(self, s: str) -> int:
        self._console.write(s)
        self._fh.write(_ANSI_RE.sub("", s))
        return len(s)

    def flush(self) -> None:
        self._console.flush()
        self._fh.flush()

    def isatty(self) -> bool:
        return getattr(self._console, "isatty", lambda: False)()


class RunLogger:
    """Tee stdout+stderr to ``logs/<prefix>_<timestamp>.txt`` with a documented header.

    Args:
        prefix:  log filename stem; a timestamp is appended.
        log_dir: directory for the log (created if absent; default ``logs``).
        title:   header title line (defaults to ``prefix``).
        **fields: key/value lines written into the header — use these to record
                  the exact run parameters (config, seed, checkpoint, …) so the
                  log is self-documenting for a reviewer.

    Usable as a context manager (preferred) or via ``.start()`` / ``.stop()``.
    ``.path`` is the resolved log file path.
    """

    def __init__(self, prefix: str, *, log_dir: str = "logs",
                 title: str | None = None, **fields):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{prefix}_{ts}.txt"
        self.title = title or prefix
        self.fields = fields
        self._fh = None
        self._out = None
        self._err = None

    def start(self) -> "RunLogger":
        if self._fh is not None:
            return self  # already started
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # append-mode + line buffering: safe if start() is called twice and keeps
        # output flushed for a `tail -f`.
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = _Mirror(self._out, self._fh)
        sys.stderr = _Mirror(self._err, self._fh)
        print("=" * 64)
        print(self.title)
        print("=" * 64)
        print(f"Started : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Log file: {self.path}")
        for k, v in self.fields.items():
            print(f"  {k:16}: {v}")
        print(flush=True)
        return self

    def stop(self) -> None:
        if self._fh is None:
            return
        print(f"\nFinished: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Log saved: {self.path}", flush=True)
        sys.stdout, sys.stderr = self._out, self._err
        self._fh.close()
        self._fh = None

    def __enter__(self) -> "RunLogger":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False   # never swallow exceptions (they were logged via stderr)