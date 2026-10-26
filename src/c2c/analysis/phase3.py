"""The Phase 3 experiments, each one cached to data/processed/phase3/<name>.npz.

``scripts/run_phase3.py`` runs them all from a terminal (the slow part, minutes to an hour);
``notebooks/03_spiking.ipynb`` calls the same functions and gets the cached results instantly.
Delete a cache file (or pass force=True) to re-run one experiment.

Graph variants used to test what our connectome choices change (see c2c.data.shiu):

    ours          our graph: >= 5 synapses, per-edge signs, modulatory edges = 0
    ours_mod      our graph, modulatory edges counted as excitatory (Shiu's convention)
    shiu_min5     Shiu's per-neuron signs, on our >= 5 synapse edge set
    shiu_full     Shiu's model input exactly: every synapse, per-neuron signs
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from c2c.analysis import spiking as sp
from c2c.data import shiu
from c2c.data.annotations import align_to_graph, load_annotations, select
from c2c.data.graph import Graph, load_graph, save_graph
from c2c.paths import PROCESSED
from c2c.sim.network import SimResult, cached, silence_columns, subnetwork, weight_matrix
from c2c.sim.params import SHIU2024
from c2c.sim.torch_lif import simulate_torch

CACHE = PROCESSED / "phase3"
VARIANTS = ["ours", "ours_mod", "shiu_min5", "shiu_full"]


@dataclass
class Network:
    """A graph variant ready to simulate: graph, annotations in matrix order, weights."""
    name: str
    g: Graph
    aligned: object
    W: object

    def sugar(self) -> np.ndarray:
        return self.g.index_of(shiu.present(self.g, shiu.SUGAR_GRNS))

    def mn9(self) -> int:
        return int(self.g.index_of([shiu.MN9])[0])

    def looming(self) -> np.ndarray:
        return select(self.aligned, cell_type=sp.LOOMING_TYPES)

    def escape(self) -> dict[str, int]:
        return sp.named_neurons(self.aligned, sp.ESCAPE_TYPES)


@cache
def _annotations():
    return load_annotations()


@cache
def network(name: str) -> Network:
    if name in ("ours", "ours_mod"):
        g = load_graph(PROCESSED / "graph_v783_min5.npz")
        W = weight_matrix(g, modulatory_sign=1 if name == "ours_mod" else 0)
    elif name in ("shiu_min5", "shiu_full"):
        path = PROCESSED / f"graph_{name}.npz"
        if path.exists():
            g = load_graph(path)
        else:
            g = shiu.load_shiu_graph(min_syn=5 if name == "shiu_min5" else 1)
            save_graph(g, path)
        W = weight_matrix(g)
    else:
        raise ValueError(f"unknown variant {name!r}; choose from {VARIANTS}")
    return Network(name, g, align_to_graph(_annotations(), g), W)


# ---- (a) sugar -> MN9 ------------------------------------------------------------------------
def sugar_sweep(variant: str, trials_per_rate: int = 10, t_run: float = 1000.0,
                force: bool = False, **kw) -> tuple[SimResult, np.ndarray]:
    """PyTorch, sugar GRNs at every rate in SUGAR_RATES, all rates in one batch."""
    net = network(variant)
    rates = sp.sweep_rates(sp.SUGAR_RATES, trials_per_rate)
    res = cached(CACHE / f"sugar_sweep_{variant}.npz", lambda: simulate_torch(
        net.W, SHIU2024, drive=net.sugar(), rate_hz=rates, t_run=t_run, n_trials=len(rates),
        record=[net.mn9()], verbose=True, **kw), force)
    return res, rates


def sugar_reference(variant: str, simulator: str, n_trials: int, rates=(100, 200),
                    t_run: float = 1000.0, force: bool = False, **kw):
    """Sugar at 100 and 200 Hz, n_trials each, in Brian2 or PyTorch (for the head-to-head)."""
    net = network(variant)
    trial_rates = sp.sweep_rates(rates, n_trials)
    path = CACHE / f"sugar_{simulator}_{variant}.npz"
    if simulator == "brian":
        from c2c.sim.brian_lif import simulate_brian

        def run():
            return simulate_brian(net.W, SHIU2024, drive=net.sugar(), rate_hz=trial_rates,
                                  t_run=t_run, n_trials=len(trial_rates), verbose=True, **kw)
    else:
        def run():
            return simulate_torch(net.W, SHIU2024, drive=net.sugar(), rate_hz=trial_rates,
                                  t_run=t_run, n_trials=len(trial_rates), record=[net.mn9()],
                                  seed=1, verbose=True, **kw)
    return cached(path, run, force), trial_rates


def local_circuit(net: Network, n_keep: int = 1500) -> np.ndarray:
    """Every neuron that fires under 200 Hz sugar drive in the full-brain run (capped at the
    n_keep most active), plus the GRNs: ~400 neurons, a real piece of connectome small enough
    for long Brian2 runs."""
    res, rates = sugar_sweep(net.name)
    m = res.rates()[:, rates == 200].mean(axis=1)
    top = np.argsort(m)[::-1][:n_keep]
    return np.union1d(top[m[top] > 0], net.sugar())


def subcircuit_comparison(variant: str = "ours", n_trials: int = 30, t_run: float = 1000.0,
                          force: bool = False):
    """Brian2 vs PyTorch, Poisson sugar drive at 200 Hz, on the sugar-driven subcircuit."""
    from c2c.sim.brian_lif import simulate_brian

    net = network(variant)
    keep = local_circuit(net)
    Wsub, keep = subnetwork(net.W, keep)
    drive = np.flatnonzero(np.isin(keep, net.sugar()))
    kw = {"drive": drive, "rate_hz": 200.0, "t_run": t_run, "n_trials": n_trials}
    rb = cached(CACHE / f"sub_brian_{variant}.npz",
                lambda: simulate_brian(Wsub, SHIU2024, seed=100, **kw), force)
    rt = cached(CACHE / f"sub_torch_{variant}.npz",
                lambda: simulate_torch(Wsub, SHIU2024, seed=200, **kw), force)
    return rb, rt, keep


# ---- (b) looming -> escape DNs -------------------------------------------------------------
def looming_sweep(variant: str = "ours", trials_per_rate: int = 20, t_run: float = 300.0,
                  force: bool = False, **kw) -> tuple[SimResult, np.ndarray]:
    net = network(variant)
    rates = sp.sweep_rates(sp.LOOM_RATES, trials_per_rate)
    res = cached(CACHE / f"looming_sweep_{variant}.npz", lambda: simulate_torch(
        net.W, SHIU2024, drive=net.looming(), rate_hz=rates, t_run=t_run, n_trials=len(rates),
        record=list(net.escape().values()), verbose=True, **kw), force)
    return res, rates


# ---- (c) silencing ---------------------------------------------------------------------------
def silencing_conditions(variant: str = "ours") -> dict[str, np.ndarray]:
    """What to silence, chosen by rule (see spiking.relay_types), plus the two input types."""
    net = network(variant)
    res, rates = looming_sweep(variant)
    intact_rates = res.rates()[:, rates == 100].mean(axis=1)
    relays = sp.relay_types(net.aligned, net.W, net.escape(), intact_rates,
                            exclude=sp.LOOMING_TYPES)
    # every presynaptic partner of the eight escape DNs except LPLC2 and LC4
    partners = np.unique(np.concatenate([net.W[i].indices for i in net.escape().values()]))
    partners = np.setdiff1d(partners, net.looming())
    conds = {
        "intact": np.array([], dtype=np.int64),
        "-LPLC2": select(net.aligned, cell_type="LPLC2"),
        "-LC4": select(net.aligned, cell_type="LC4"),
        "-relays": select(net.aligned, cell_type=relays),
        "direct only": partners,
    }
    return conds, relays


def silencing(variant: str = "ours", rates=(50, 100), trials_per_rate: int = 20,
              t_run: float = 300.0, force: bool = False):
    net = network(variant)
    conds, relays = silencing_conditions(variant)
    trial_rates = sp.sweep_rates(rates, trials_per_rate)
    out = {}
    for name, idx in conds.items():
        slug = name.replace(" ", "_").replace("-", "minus_")
        Wc = silence_columns(net.W, idx) if len(idx) else net.W
        out[name] = cached(CACHE / f"silence_{variant}_{slug}.npz", lambda Wc=Wc: simulate_torch(
            Wc, SHIU2024, drive=net.looming(), rate_hz=trial_rates, t_run=t_run,
            n_trials=len(trial_rates), record=list(net.escape().values()), seed=7), force)
    return out, trial_rates, conds, relays
