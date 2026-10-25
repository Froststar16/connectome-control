"""Figures for Phase 2. Kept apart from pathways.py so the algorithms import no matplotlib and
stay quick to test; styling reuses the Phase 1 palette from graph_stats."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from c2c.analysis.graph_stats import EXC, INH, INK, INK_2, MOD, SEQ_BLUE, SURFACE

SIGN_COLOR = {1: EXC, -1: INH, 0: MOD}


def layered_positions(layer: dict[str, int]) -> dict[str, tuple[float, float]]:
    """x = layer, y = evenly spaced within the layer (alphabetical, so the layout is stable)."""
    pos = {}
    for x in sorted(set(layer.values())):
        col = sorted(n for n, lv in layer.items() if lv == x)
        ys = np.linspace(1, -1, len(col)) if len(col) > 1 else [0.0]
        pos.update({n: (float(x), float(y)) for n, y in zip(col, ys)})
    return pos


def plot_layered_circuit(layer: dict[str, int], edges: pd.DataFrame, ax=None,
                         highlight: set[str] = frozenset(), layer_names: list[str] | None = None,
                         max_width: float = 7.0):
    """Cell-type circuit drawn left -> right by layer.

    ``edges`` needs columns pre, post, fraction (share of post's input), sign.
    Line width is proportional to the fraction, colour to the sign.
    """
    ax = ax or plt.subplots(figsize=(12, 6))[1]
    pos = layered_positions(layer)
    fmax = edges["fraction"].max()
    for e in edges.sort_values("fraction").itertuples():
        (x0, y0), (x1, y1) = pos[e.pre], pos[e.post]
        rad = 0.25 if x1 <= x0 else 0.08          # bend backward / same-layer edges more
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=8, lw=0.6 + max_width * e.fraction / fmax,
            color=SIGN_COLOR[int(e.sign)], alpha=0.75, shrinkA=22, shrinkB=22, zorder=1))
    for name, (x, y) in pos.items():
        bold = name in highlight
        ax.text(x, y, name, ha="center", va="center", zorder=3,
                fontsize=9, fontweight="bold" if bold else "normal", color=INK,
                bbox={"boxstyle": "round,pad=0.35", "fc": SURFACE,
                      "ec": INK if bold else INK_2, "lw": 1.4 if bold else 0.7})
    xs = sorted(set(layer.values()))
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([])
    ax.set_xticks(xs, layer_names or [f"layer {x}" for x in xs])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    handles = [plt.Line2D([], [], color=c, lw=3, label=lbl) for c, lbl in
               [(EXC, "excitatory"), (INH, "inhibitory"), (MOD, "modulatory / mixed")]]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, fontsize=8)
    return ax


def plot_hop_histogram(dist_all: np.ndarray, dist_target: np.ndarray, target_label: str,
                       ax=None):
    """Share of neurons at each hop distance, all neurons vs a target class (e.g. DNs)."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    hops = np.arange(1, max(dist_all.max(), dist_target.max()) + 1)

    def share(d):
        return np.array([(d == h).mean() for h in hops])

    visible = (share(dist_all) >= 1e-3) | (share(dist_target) >= 1e-3)
    hops = hops[: np.flatnonzero(visible).max() + 1]   # drop empty trailing bars
    w = 0.4
    ax.bar(hops - w / 2, share(dist_all), w, color=MOD, label="all neurons")
    ax.bar(hops + w / 2, share(dist_target), w, color=EXC, label=target_label)
    ax.set_xticks(hops)
    ax.set_xlabel("hops from the nearest sensory neuron")
    ax.set_ylabel("share of neurons")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    return ax


def plot_modality_heatmap(table: pd.DataFrame, ax=None, fmt: str = "{:.0%}"):
    """Rows = modalities, columns = hop counts, cells = share of DNs first reached at that hop."""
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for (r, c), v in np.ndenumerate(data):
        if v >= 0.005:
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8,
                    color=SURFACE if v > 0.5 else INK)
    ax.set_xlabel("hops to the descending neuron")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.8, label="share of DNs")
    return ax


def plot_layer_by_group(layer: pd.Series, groups: pd.Series, order: list[str], ax=None,
                        xlabel: str = "information-flow layer"):
    """Horizontal box plot of a per-neuron layer, one box per group."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    data = [layer[groups == g_].dropna().to_numpy() for g_ in order]
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=order, showfliers=False,
                    patch_artist=True, widths=0.6)
    for box in bp["boxes"]:
        box.set(facecolor="#cde2fb", edgecolor=INK_2)
    for med in bp["medians"]:
        med.set(color=EXC, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    return ax
