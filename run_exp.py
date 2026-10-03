#!/usr/bin/env python3
"""Thin local process wrapper for long-running training / eval commands.

Keeps stdio attached and forwards SIGINT/SIGTERM to the child process group.
Useful when you want a stable parent process around ``run_agent.sh`` or a
custom shell pipeline. No experiment-tracking side effects.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(
            "usage: run_exp.py CMD [ARGS...]\n"
            '       run_exp.py --shell "CMD | WITH > operators"',
            file=sys.stderr,
        )
        return 2

    if argv[0] == "--shell":
        if len(argv) != 2:
            print(
                '--shell requires exactly one quoted command string, e.g.:\n'
                '    run_exp.py --shell "train | tee out.log"',
                file=sys.stderr,
            )
            return 2
        cmd: str | list[str] = argv[1]
        use_shell = True
    else:
        cmd = argv
        use_shell = False

    popen_kwargs: dict = {"start_new_session": True}
    if use_shell:
        popen_kwargs["executable"] = "/bin/bash"
    proc = subprocess.Popen(cmd, shell=use_shell, **popen_kwargs)

    def _forward(signum, _frame):
        fwd = signal.SIGTERM if signum == signal.SIGHUP else signum
        try:
            os.killpg(proc.pid, fwd)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _forward)

    rc = proc.wait()
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    raise SystemExit(main())
