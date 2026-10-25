"""Load the FlyWire v783 connectivity table and collapse it to neuron-to-neuron edges.

The raw file has one row per (pre, post, neuropil). We sum synapses over neuropils
so a connection split across brain regions is counted once, then apply a threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"

PRE, POST = "pre_pt_root_id", "post_pt_root_id"
NT_COLS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
NT_NAMES = {c: c.removesuffix("_avg").upper() for c in NT_COLS}  # gaba_avg -> GABA


def load_raw_connections(path: Path | None = None) -> pd.DataFrame:
    """Read proofread_connections_783.feather (one row per pair per neuropil)."""
    return pd.read_feather(path or RAW_DIR / "proofread_connections_783.feather")


def load_proofread_ids(path: Path | None = None) -> np.ndarray:
    return np.load(path or RAW_DIR / "proofread_root_ids_783.npy")


def collapse_neuropils(raw: pd.DataFrame) -> pd.DataFrame:
    """Sum syn_count over neuropils for each pre->post pair.

    Neurotransmitter probabilities are averaged weighted by synapse count,
    so a neuropil with 50 synapses counts more than one with 2.
    """
    df = raw[[PRE, POST, "syn_count", *NT_COLS]].copy()
    for c in NT_COLS:
        df[c] = df[c] * df["syn_count"]
    edges = df.groupby([PRE, POST], as_index=False, sort=False).sum()
    for c in NT_COLS:
        edges[c] = edges[c] / edges["syn_count"]
    edges["nt_type"] = edges[NT_COLS].idxmax(axis=1).map(NT_NAMES)
    return edges


def threshold(edges: pd.DataFrame, min_syn: int = 5) -> pd.DataFrame:
    """Keep pairs with at least `min_syn` synapses (5 is the common FlyWire convention)."""
    return edges[edges["syn_count"] >= min_syn].reset_index(drop=True)


def summarize(edges_all: pd.DataFrame, edges_thr: pd.DataFrame, min_syn: int) -> dict:
    neurons = pd.unique(edges_thr[[PRE, POST]].to_numpy().ravel())
    nt_share = edges_thr.groupby("nt_type")["syn_count"].sum() / edges_thr["syn_count"].sum()
    return {
        "pairs_before_threshold": len(edges_all),
        f"pairs_with_>={min_syn}_synapses": len(edges_thr),
        "neurons_in_thresholded_graph": len(neurons),
        "total_synapses_after_threshold": int(edges_thr["syn_count"].sum()),
        "nt_share_by_synapses": nt_share.sort_values(ascending=False).round(3).to_dict(),
    }


def main(min_syn: int = 5) -> None:
    ids = load_proofread_ids()
    print(f"Proofread neurons (npy): {len(ids):,}")
    raw = load_raw_connections()
    print(f"Raw rows (pair x neuropil): {len(raw):,}")
    edges = collapse_neuropils(raw)
    thr = threshold(edges, min_syn)
    for k, v in summarize(edges, thr, min_syn).items():
        if isinstance(v, int):
            v = f"{v:,}"
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
