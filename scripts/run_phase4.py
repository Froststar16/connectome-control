"""Run (and cache) every Phase 4 simulation. The notebook then just loads the results.

Usage (repo root, .venv active, Phase 3 data in place):
    python scripts/run_phase4.py                       # everything on shiu_min5, 3 seeds
    python scripts/run_phase4.py --variant ours        # same on our per-edge-sign graph
    python scripts/run_phase4.py --seeds 0             # quick: one seed per condition
    python scripts/run_phase4.py --factorial           # also the giant-fiber question

Each closed-loop run simulates 1 s of brain + body + eyes; on a laptop CPU that takes a few
minutes. Results land in data/processed/phase4/*.npz, the escape GIF in results/figures/.
"""

from __future__ import annotations

import argparse
import time

from c2c.analysis import phase4 as p4
from c2c.paths import FIGURES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="shiu_min5", help="graph variant (see phase3.VARIANTS)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--factorial", action="store_true",
                    help="looming sweep on all four variants (which choice silences the GF?)")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    args = ap.parse_args()
    t0 = time.perf_counter()
    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.factorial:
        print("=== giant-fiber factorial (Phase 3 looming sweep on each variant)")
        print(p4.giant_fiber_factorial().round(1).to_string(index=False))

    print(f"\n=== closed-loop experiments, variant {args.variant}, seeds {args.seeds}")
    out = p4.experiments(args.variant, seeds=args.seeds, force=args.force,
                         gif=FIGURES / f"phase4_escape_{args.variant}.gif")
    for cond, logs in out.items():
        for log in logs:
            s = p4.summary(log)
            print(f"{cond:14s} backward cmd at {s['backward_cmd_ms']:6.1f} ms, "
                  f"closest gap {s['min_gap_mm']:5.2f} mm, final x {s['x_final_mm']:+6.2f} mm")
    print(f"\ndone in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
