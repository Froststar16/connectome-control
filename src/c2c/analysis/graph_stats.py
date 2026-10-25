"""Descriptive statistics of the connectome graph, and the figures that show them.

Each statistic is a pure function of a Graph that returns numbers (easy to test);
each ``plot_*`` function takes those numbers and an Axes (easy to restyle / combine).
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import sparse

from c2c.data.graph import Graph

# ---- colours (one blue for excitation, one orange for inhibition, grey for modulatory) ----
EXC = "#2a78d6"
INH = "#eb6834"
MOD = "#b8b6ae"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
# Single-hue sequential ramp (light -> dark blue) for magnitudes such as the heatmap.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def apply_style() -> None:
    """Quiet axes so the data is the loudest thing on the figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


# ========================================================================================
# Statistics
# ========================================================================================
def in_degree(g: Graph) -> np.ndarray:
    """Number of presynaptic partners of each neuron = stored entries in its ROW.

    In CSR, row i occupies data[indptr[i]:indptr[i+1]], so the row lengths are diff(indptr).
    """
    return np.diff(g.counts.indptr)


def out_degree(g: Graph) -> np.ndarray:
    """Number of postsynaptic partners = how often each column index appears in ``indices``."""
    return np.bincount(g.counts.indices, minlength=g.n)


def log_binned_hist(values: np.ndarray, n_bins: int = 30):
    """Histogram with logarithmically spaced bins, normalised to a probability density.

    Equal-width bins are useless for heavy-tailed data: almost everything lands in the first
    bin and the tail is single counts. Log bins give each decade the same number of bins, and
    dividing by bin width turns counts into a density that is comparable across bins.
    Zeros are dropped (log(0) is undefined); callers should report how many there were.
    """
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    edges = np.logspace(np.log10(v.min()), np.log10(v.max() + 1), n_bins + 1)
    counts, edges = np.histogram(v, bins=edges)
    density = counts / (np.diff(edges) * len(v))
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean = centre on a log axis
    keep = counts > 0
    return centers[keep], density[keep]


def merge_hemispheres(labels: np.ndarray) -> np.ndarray:
    """'AL_L', 'AL_R' -> 'AL'. Midline neuropils without a side suffix are unchanged."""
    return np.array([re.sub(r"_(L|R)$", "", s) for s in labels])


def region_labels(g: Graph, merge: bool = True) -> np.ndarray:
    return merge_hemispheres(g.home_neuropil) if merge else g.home_neuropil


def neuropil_matrix(g: Graph, merge: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Synapses between regions, where a neuron belongs to its home neuropil.

    Returns (M, names) with M[a, b] = synapses from neurons homed in region a (source)
    onto neurons homed in region b (target), regions ordered by total synapses.

    How: build a one-hot membership matrix P (n x k), P[i, a] = 1 if neuron i is in region a.
    Then (P.T @ counts @ P)[b, a] sums counts[i, j] over all targets i in b and sources j in a.
    Transposing gives source-by-target. Two sparse products instead of a Python loop.
    """
    labels = region_labels(g, merge)
    codes, names = pd.factorize(labels, sort=True)  # integer region code per neuron
    k = len(names)
    P = sparse.csr_matrix(
        (np.ones(g.n, dtype=np.int64), (np.arange(g.n), codes)), shape=(g.n, k)
    )
    counts64 = g.counts.astype(np.int64)  # avoid int32 overflow when summing millions
    M = (P.T @ counts64 @ P).T.toarray()

    order = np.argsort(-(M.sum(0) + M.sum(1)), kind="stable")  # biggest regions first
    return M[np.ix_(order, order)], np.asarray(names)[order]


def ei_share_by_region(g: Graph, merge: bool = True) -> pd.DataFrame:
    """Excitatory / inhibitory / modulatory share of the INPUT synapses of each region.

    Every stored edge is an input to its row neuron i; we tag it with i's home region and the
    sign of the edge, then sum synapse counts per (region, sign).
    """
    rows = np.repeat(np.arange(g.n), np.diff(g.counts.indptr))  # row index of every edge
    region = region_labels(g, merge)[rows]
    df = pd.DataFrame({"region": region, "sign": g.sign, "syn": g.counts.data})
    table = df.pivot_table(index="region", columns="sign", values="syn", aggfunc="sum",
                           fill_value=0)
    table = table.reindex(columns=[1, -1, 0], fill_value=0)
    table.columns = ["excitatory", "inhibitory", "modulatory"]
    table["total"] = table.sum(axis=1)
    for c in ["excitatory", "inhibitory", "modulatory"]:
        table[f"{c}_share"] = table[c] / table["total"]
    return table.sort_values("excitatory_share", ascending=False)


def summary(g: Graph) -> dict:
    """Headline numbers for the README / notebook."""
    W = g.W
    kin, kout = in_degree(g), out_degree(g)
    return {
        "neurons": g.n,
        "edges": g.counts.nnz,
        "synapses": int(g.counts.sum()),
        "density": g.counts.nnz / g.n**2,
        "mean_in_degree": kin.mean(),
        "max_in_degree": int(kin.max()),
        "max_out_degree": int(kout.max()),
        "neurons_no_input": int((kin == 0).sum()),
        "neurons_no_output": int((kout == 0).sum()),
        "excitatory_synapse_share": W.data[W.data > 0].sum() / g.counts.sum(),
        "inhibitory_synapse_share": -W.data[W.data < 0].sum() / g.counts.sum(),
    }


# ========================================================================================
# Figures
# ========================================================================================
def plot_degree_distributions(g: Graph, ax=None):
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    for deg, name, color in [(in_degree(g), "in-degree", EXC), (out_degree(g), "out-degree", INH)]:
        x, y = log_binned_hist(deg)
        ax.loglog(x, y, "-o", color=color, ms=4, mec=SURFACE, mew=1, label=name)
    ax.set_xlabel("partners per neuron (degree)")
    ax.set_ylabel("probability density")
    ax.set_title("Degree distributions (≥5-synapse edges)")
    ax.legend(frameon=False)
    return ax


def plot_weight_distribution(g: Graph, min_syn: int = 5, ax=None):
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    x, y = log_binned_hist(g.counts.data)
    ax.loglog(x, y, "-o", color=EXC, ms=4, mec=SURFACE, mew=1)
    ax.axvline(min_syn, color=INK_2, lw=1, ls="--")
    ax.annotate(f"threshold = {min_syn}", (min_syn, y.max()), xytext=(6, -2),
                textcoords="offset points", color=INK_2, fontsize=9, va="top")
    ax.set_xlabel("synapses per connection")
    ax.set_ylabel("probability density")
    ax.set_title("Connection-weight distribution")
    return ax


def plot_neuropil_heatmap(M: np.ndarray, names: np.ndarray, top: int | None = None,
                          normalize: str = "none", ax=None):
    """normalize='none': raw synapse counts; 'target': each column sums to 1, i.e. the fraction
    of a target region's input that comes from each source region."""
    if top:
        M, names = M[:top, :top], names[:top]
    M = M.astype(float)
    if normalize == "target":
        M = M / np.maximum(M.sum(axis=0, keepdims=True), 1)
    ax = ax or plt.subplots(figsize=(9, 8))[1]
    data = np.ma.masked_less_equal(M, 0)  # empty cells drawn as background, not dark blue
    vmin = data.min() if normalize == "none" else 1e-4
    im = ax.imshow(data, cmap=SEQ_BLUE, norm=LogNorm(vmin=vmin, vmax=data.max()),
                   interpolation="nearest")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("target region (postsynaptic home neuropil)")
    ax.set_ylabel("source region (presynaptic home neuropil)")
    ax.grid(False)
    label = "synapses" if normalize == "none" else "fraction of target's input"
    plt.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set_title("Region-to-region connectivity")
    return ax


def plot_ei_share(table: pd.DataFrame, min_synapses: int = 10_000, ax=None):
    t = table[table["total"] >= min_synapses].sort_values("excitatory_share")
    ax = ax or plt.subplots(figsize=(6, max(4, 0.2 * len(t))))[1]
    left = np.zeros(len(t))
    ax.set_axisbelow(True)  # draw gridlines behind the bars
    for col, color, label in [("excitatory_share", EXC, "excitatory (ACH)"),
                              ("inhibitory_share", INH, "inhibitory (GABA/GLUT)"),
                              ("modulatory_share", MOD, "modulatory")]:
        ax.barh(t.index, t[col], left=left, color=color, edgecolor=SURFACE, linewidth=1,
                height=0.8, label=label)
        left += t[col].to_numpy()
    ax.axvline(0.5, color=INK_2, lw=0.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of input synapses")
    ax.set_title("Excitation vs inhibition by region", pad=24)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    # Legend in one row between the title and the plot, so it never hits the x-axis label.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8,
              borderaxespad=0.2, handlelength=1.2, columnspacing=1.0)
    return ax
