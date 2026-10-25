"""Build the signed connectome matrix from the raw FlyWire table and save it.

Usage (from the repo root, with the .venv active):
    python scripts/build_graph.py            # default threshold 5 synapses
    python scripts/build_graph.py --min-syn 10
"""

import argparse
import time

import numpy as np
import pandas as pd

from c2c.data.graph import build_graph, save_graph
from c2c.data.loaders import NT_COLS, POST, PRE, collapse_neuropils, threshold
from c2c.paths import PROCESSED, RAW

RAW_FILE = RAW / "proofread_connections_783.feather"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-syn", type=int, default=5)
    args = parser.parse_args()
    t0 = time.perf_counter()

    cols = [PRE, POST, "neuropil", "syn_count", *NT_COLS]
    raw = pd.read_feather(RAW_FILE, columns=cols)
    raw["neuropil"] = raw["neuropil"].astype("category")  # ~80 names repeated 16.8M times
    print(f"raw rows:            {len(raw):,}")

    edges = threshold(collapse_neuropils(raw), min_syn=args.min_syn)
    print(f"edges >= {args.min_syn} syn:      {len(edges):,}")

    g = build_graph(edges, raw)
    W = g.W
    print(f"neurons (n):         {g.n:,}")
    print(f"counts nnz:          {g.counts.nnz:,}")
    print(f"total synapses:      {int(g.counts.sum()):,}")
    print(f"W nnz (signed):      {W.nnz:,}")
    print(f"excitatory synapses: {int(W.data[W.data > 0].sum()):,}")
    print(f"inhibitory synapses: {int(-W.data[W.data < 0].sum()):,}")
    print(f"distinct neuropils:  {len(np.unique(g.home_neuropil))}")

    out = save_graph(g, PROCESSED / f"graph_v783_min{args.min_syn}.npz")
    print(f"saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
