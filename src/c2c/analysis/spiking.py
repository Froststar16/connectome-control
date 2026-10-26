"""Phase 3 experiments on the spiking model: stimulus sweeps, readout tables, silencing, plots.

The simulators (c2c.sim) know nothing about flies; this module is where cell types come in.
Every experiment is "drive these neurons with Poisson input, read out those neurons", run as ONE
batched simulation where each trial (column) can have its own stimulus rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from c2c.data.annotations import select
from c2c.sim.network import SimResult

ESCAPE_TYPES = ["DNp01", "DNp04", "DNp02", "DNp11"]   # giant fiber first, then Phase 2 order
LOOMING_TYPES = ["LPLC2", "LC4"]
SUGAR_RATES = [10, 25, 50, 75, 100, 125, 150, 175, 200]
LOOM_RATES = [10, 20, 50, 100, 150, 200]


# ---- naming neurons -----------------------------------------------------------------------
def named_neurons(aligned: pd.DataFrame, cell_types, by_side: bool = True) -> dict[str, int]:
    """{"DNp01 L": index, ...} for cell types that have one neuron per side."""
    out = {}
    for t in cell_types:
        sides = ["left", "right"] if by_side else [None]
        for s in sides:
            idx = select(aligned, cell_type=t, side=s) if s else select(aligned, cell_type=t)
            if len(idx) != 1:
                raise ValueError(f"expected one {t} on side {s}, found {len(idx)}")
            out[f"{t} {s[0].upper()}" if s else t] = int(idx[0])
    return out


# ---- sweeps -------------------------------------------------------------------------------
def sweep_rates(rates, trials_per_rate: int) -> np.ndarray:
    """Per-trial stimulus rates for a batched sweep: [r0]*n + [r1]*n + ..."""
    return np.repeat(np.asarray(rates, dtype=float), trials_per_rate)


def sweep_table(res: SimResult, trial_rates, neurons: dict[str, int]) -> pd.DataFrame:
    """One row per (neuron, stimulus rate): mean rate, SEM, P(fired), median first-spike time.

    Median latency is taken over the trials in which the neuron fired, so read it together with
    p_fire: a 40 ms latency in 1 of 10 trials is noise, in 10 of 10 it is a response.
    """
    trial_rates = np.asarray(trial_rates)
    rows = []
    rates = res.rates()
    for name, i in neurons.items():
        for r_in in np.unique(trial_rates):
            cols = trial_rates == r_in
            fr = rates[i, cols]
            fs = res.first_spike[i, cols]
            fired = ~np.isnan(fs)
            rows.append({
                "neuron": name, "input_hz": r_in,
                "rate_hz": fr.mean(), "sem_hz": fr.std(ddof=1) / np.sqrt(cols.sum()),
                "p_fire": fired.mean(),
                "latency_ms": np.median(fs[fired]) if fired.any() else np.nan,
            })
    return pd.DataFrame(rows)


def rate_agreement(a: np.ndarray, b: np.ndarray, min_rate: float = 1.0) -> dict:
    """How well two per-neuron rate vectors agree, on neurons active (>= min_rate) in either."""
    active = (a >= min_rate) | (b >= min_rate)
    x, y = a[active], b[active]
    slope = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else np.nan
    return {
        "active_either": int(active.sum()),
        "active_both": int(((a >= min_rate) & (b >= min_rate)).sum()),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]) if active.sum() > 2 else np.nan,
        "slope_b_vs_a": slope,
        "median_abs_diff_hz": float(np.median(np.abs(x - y))) if active.any() else np.nan,
    }


def z_scores(a_mean, a_sem, b_mean, b_sem, floor: float = 0.5) -> np.ndarray:
    """(a - b) / combined standard error, with a small floor for neurons that never fire."""
    return (a_mean - b_mean) / (np.sqrt(a_sem**2 + b_sem**2) + floor)


# ---- silencing -----------------------------------------------------------------------------
def relay_types(aligned, W, targets: dict[str, int], rates: np.ndarray, exclude,
                min_fraction: float = 0.02, min_rate: float = 10.0) -> list[str]:
    """Cell types that (a) fire >= min_rate Hz in the intact run and (b) provide >= min_fraction
    of the input synapses of at least one target. These are the candidate relays of the signal,
    chosen by rule rather than by eye."""
    absW = abs(W)
    total = np.asarray(absW.sum(axis=1)).ravel()
    ct = aligned["cell_type"].to_numpy(dtype=object)
    found = set()
    for i in targets.values():
        row = absW[i]
        frac = pd.Series(row.data / total[i]).groupby(ct[row.indices]).sum()
        active = pd.Series(rates[row.indices]).groupby(ct[row.indices]).mean()
        keep = frac[(frac >= min_fraction) & (active.reindex(frac.index) >= min_rate)].index
        found |= set(keep)
    return sorted(found - set(exclude))


# ---- plots ---------------------------------------------------------------------------------
def plot_sweep(table: pd.DataFrame, value: str = "rate_hz", ax=None, err: str | None = "sem_hz",
               title: str = "", ylabel: str | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    # one colour per cell type, solid = left, dashed = right ("DNp01 L" / "DNp01 R")
    types = list(dict.fromkeys(n.split(" ")[0] for n in table["neuron"]))
    colours = {t: f"C{k}" for k, t in enumerate(types)}
    for name, d in table.groupby("neuron", sort=False):
        d = d.sort_values("input_hz")
        y = d[value].to_numpy()
        style = {"color": colours[name.split(" ")[0]], "marker": "o", "ms": 4,
                 "ls": "--" if name.endswith(" R") else "-", "label": name}
        if err:
            ax.errorbar(d["input_hz"], y, yerr=d[err], capsize=2, **style)
        else:
            ax.plot(d["input_hz"], y, **style)
    ax.set_xlabel("input rate per stimulated neuron (Hz)")
    ax.set_ylabel(ylabel or value)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    return ax


def plot_raster(res: SimResult, neurons: dict[str, int], trial: int = 0, ax=None,
                t_max: float | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    r = res.raster
    for row, (name, i) in enumerate(neurons.items()):
        t = r["t"][(r["neuron"] == i) & (r["trial"] == trial)]
        ax.vlines(t, row + 0.6, row + 1.4, lw=0.8)
    ax.set_yticks(range(1, len(neurons) + 1), list(neurons))
    ax.set_xlabel("time (ms)")
    ax.set_xlim(0, t_max or res.t_run)
    ax.invert_yaxis()
    return ax
