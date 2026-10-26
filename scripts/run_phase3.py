"""Run (and cache) every Phase 3 simulation. The notebook then just loads the results.

Usage (repo root, .venv active):
    python scripts/download_data.py --shiu   # once: Shiu et al.'s connectivity table
    python scripts/run_phase3.py             # everything (Brian2 part can take ~1 h on numpy)
    python scripts/run_phase3.py --skip-brian
    python scripts/run_phase3.py --brian-trials 30 --codegen cython   # paper-sized reference

Results land in data/processed/phase3/*.npz. Delete a file to re-run that experiment.
"""

from __future__ import annotations

import argparse
import time

from c2c.analysis import phase3 as p3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brian-trials", type=int, default=10,
                    help="trials per rate for the whole-brain Brian2 reference (paper: 30)")
    ap.add_argument("--codegen", default="auto", choices=["auto", "cython", "numpy"],
                    help="Brian2 code generation target (cython needs a C++ compiler)")
    ap.add_argument("--skip-brian", action="store_true")
    ap.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    args = ap.parse_args()
    tkw = {"device": args.device} if args.device else {}
    t0 = time.perf_counter()

    def step(label):
        print(f"\n=== {label}  ({time.perf_counter() - t0:.0f} s elapsed)")

    for v in p3.VARIANTS:
        step(f"(a) sugar sweep, PyTorch, {v}")
        p3.sugar_sweep(v, **tkw)
    for v in ("ours", "shiu_full"):
        step(f"(a) sugar 100/200 Hz, PyTorch x30, {v}")
        p3.sugar_reference(v, "torch", n_trials=30, **tkw)
    if not args.skip_brian:
        for v in ("ours", "shiu_full"):
            step(f"(a) sugar 100/200 Hz, Brian2 x{args.brian_trials}, {v}")
            p3.sugar_reference(v, "brian", n_trials=args.brian_trials, codegen=args.codegen)
        step("(a) Brian2 vs PyTorch on the sugar subcircuit")
        p3.subcircuit_comparison()
    for v in ("ours", "shiu_full"):
        step(f"(b) looming sweep, {v}")
        p3.looming_sweep(v, **tkw)
    step("(c) silencing")
    p3.silencing()
    step("done")


if __name__ == "__main__":
    main()
