"""Run (and cache) every Phase 5 computation. The notebook then just loads the results.

Usage (repo root, .venv active, data from `python scripts/download_data.py --minimal`):
    python scripts/run_phase5.py                 # everything: 5 azimuths x 2 encoders x 3 seeds
    python scripts/run_phase5.py --seeds 0       # quick: one seed per condition (~30 min)
    python scripts/run_phase5.py --no-gif        # skip the rendered top-view GIF

Each closed-loop run simulates 1 s of eyes + brain + body, about 1-2 minutes on a laptop CPU
(rendering the eyes is most of it), so the full set (30 closed-loop runs, 12 recordings, the
replays and the GIF) takes roughly 1-2 hours. Results land in data/processed/phase5/,
the GIF in results/figures/.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from c2c.analysis import phase5 as p5
from c2c.paths import FIGURES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(p5.SEEDS))
    ap.add_argument("--azimuths", type=float, nargs="+", default=list(p5.AZIMUTHS))
    ap.add_argument("--no-gif", action="store_true", help="skip the rendered GIF run")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    args = ap.parse_args()
    t0 = time.perf_counter()
    FIGURES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    print("=== receptive fields from the connectome")
    rf, diag = p5.receptive_field_table(force=args.force)
    print(diag)
    print(rf.groupby(["cell_type", "side"])[["az", "el", "sigma"]].median().round(1))

    print("\n=== Dombrovski et al. 2023: LC -> DN synapse gradients")
    grid = p5.lc_dn_synapses(force=args.force)
    for ct, dn, axis in [("LC4", "DNp02", "az_abs"), ("LC4", "DNp11", "az_abs"),
                         ("LPLC2", "DNp01", "el")]:
        g = p5.gradient(grid, ct, dn, axis)
        print(f"{ct:5s} -> {dn}: {g['slope_per_10deg']:+.2f} synapses per 10 deg of {axis}, "
              f"r = {g['r']:+.2f}, p = {g['p']:.1e}")

    print("\n=== steering calibration (body only)")
    print(p5.steering_calibration(force=args.force).round(1).to_string(index=False))

    print("\n=== standing-fly recordings, replayed into the brain alone (10 trials each)")
    record = p5.record_drive(force=args.force)
    replay = p5.replay_table(record, force=args.force)
    print(replay.groupby(["encoder", "azimuth", "dn"])["spikes"].mean().unstack("dn").round(1))

    print("\n=== closed-loop azimuth sweep")
    runs = p5.experiments(args.azimuths, seeds=args.seeds, force=args.force)
    table = p5.summary_table(runs)
    cols = ["encoder", "azimuth", "seed", "turn_deg", "escape_dir_deg", "backed_up", "away"]
    print(table[cols].round(1).to_string(index=False))

    if not args.no_gif:
        print("\n=== top-view GIF of a side approach")
        p5.gif_run(force=args.force)
    print(f"\ndone in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
