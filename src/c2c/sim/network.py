"""Glue between the Graph (Phase 1) and the two simulators: weight matrix, silencing, results.

Both simulators take the same inputs and return the same ``SimResult``, so every experiment can
run on either one and be compared directly.

Weight matrix
-------------
The simulators want a signed synapse-count matrix with the Graph's orientation
(rows = targets, columns = sources). ``Graph.W`` already is that, with one choice to make: what
to do with modulatory edges (serotonin, dopamine, octopamine), which Phase 1 gave sign 0.

* ``modulatory_sign=0``  (ours) drop them: they do not carry fast excitation or inhibition.
* ``modulatory_sign=+1`` (Shiu et al.) treat them as excitatory. Their model only calls GABA and
  glutamate inhibitory, so every other transmitter, modulators included, ends up excitatory.

Silencing
---------
Like Shiu et al., a silenced neuron keeps its own dynamics but all of its OUTGOING synapses are
set to zero (column j of W). It may still spike if driven, but nobody hears it. Zeroing columns
rather than deleting neurons keeps every index the same, so results line up with the intact run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from c2c.data.graph import Graph


def weight_matrix(g: Graph, modulatory_sign: int = 0, silence=None) -> sparse.csr_matrix:
    """Signed synapse counts (float64 CSR, rows = targets) ready for simulation."""
    if modulatory_sign not in (0, 1, -1):
        raise ValueError("modulatory_sign must be 0, +1 or -1")
    sign = g.sign.astype(np.int32)
    sign[sign == 0] = modulatory_sign
    W = g.counts.astype(np.float64)       # copy; same storage order as g.sign
    W.data *= sign
    W.eliminate_zeros()
    if silence is not None:
        W = silence_columns(W, silence)
    return W


def silence_columns(W: sparse.csr_matrix, idx) -> sparse.csr_matrix:
    """Copy of W with every outgoing synapse of the neurons in ``idx`` set to zero."""
    idx = np.asarray(idx, dtype=np.int64).ravel()
    W = W.copy()
    muted = np.zeros(W.shape[1], dtype=bool)
    muted[idx] = True
    W.data[muted[W.indices]] = 0.0        # W.indices = source column of each stored edge
    W.eliminate_zeros()
    return W


def edge_list(W: sparse.csr_matrix):
    """(pre, post, weight) arrays of every stored edge. CSR row = post, column = pre."""
    W = W.tocsr()
    post = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    return W.indices.astype(np.int64), post.astype(np.int64), W.data.astype(np.float64)


def subnetwork(W: sparse.csr_matrix, keep) -> tuple[sparse.csr_matrix, np.ndarray]:
    """W restricted to the neurons in ``keep`` (sorted), plus the kept original indices.
    Inputs from neurons outside ``keep`` are simply gone, so this is a smaller, different network,
    not a window onto the big one. Useful for fast Brian2-vs-PyTorch comparisons."""
    keep = np.unique(np.asarray(keep, dtype=np.int64))
    return W[keep][:, keep].tocsr(), keep


@dataclass
class SimResult:
    """Spikes from a batch of independent trials.

    counts      (n_neurons, n_trials) int   number of spikes per neuron per trial
    first_spike (n_neurons, n_trials) float time of first spike in ms, NaN if it never fired
    t_run       trial duration in ms
    raster      optional (trial, neuron, t_ms) arrays for the neurons that were recorded
    v_trace     optional (n_steps, n_recorded, n_trials) membrane potential in mV
    """

    counts: np.ndarray
    first_spike: np.ndarray
    t_run: float
    raster: dict | None = None
    v_trace: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return self.counts.shape[1]

    def rates(self) -> np.ndarray:
        """Firing rate in Hz of every neuron in every trial, (n_neurons, n_trials)."""
        return self.counts / (self.t_run / 1000.0)

    def mean_rate(self) -> np.ndarray:
        return self.rates().mean(axis=1)

    def sem_rate(self) -> np.ndarray:
        """Standard error of the mean rate across trials."""
        return self.rates().std(axis=1, ddof=1) / np.sqrt(self.n_trials)

    def latency(self) -> np.ndarray:
        """Median first-spike time over the trials in which the neuron fired (NaN if never)."""
        fs = self.first_spike
        out = np.full(fs.shape[0], np.nan)
        fired = ~np.isnan(fs).all(axis=1)
        out[fired] = np.nanmedian(fs[fired], axis=1)
        return out

    def p_fire(self) -> np.ndarray:
        """Fraction of trials in which each neuron fired at least once."""
        return (self.counts > 0).mean(axis=1)


def per_trial_rates(rate_hz, n_drive: int, n_trials: int) -> np.ndarray:
    """Normalise a stimulus rate to shape (n_drive, n_trials).

    Accepts a scalar (same rate everywhere), a (n_trials,) vector (one rate per trial, which is
    how a rate sweep runs as ONE batched simulation) or a full (n_drive, n_trials) array.
    """
    r = np.asarray(rate_hz, dtype=np.float64)
    if r.ndim == 0:
        return np.full((n_drive, n_trials), float(r))
    if r.ndim == 1:
        if r.shape[0] != n_trials:
            raise ValueError(f"got {r.shape[0]} per-trial rates for {n_trials} trials")
        return np.broadcast_to(r, (n_drive, n_trials)).copy()
    if r.shape != (n_drive, n_trials):
        raise ValueError(f"rate array must be ({n_drive}, {n_trials}), got {r.shape}")
    return r.copy()


# ----------------------------------------------------------------------------------------
# Saving results (whole-brain runs take minutes, so experiments cache them to disk)
# ----------------------------------------------------------------------------------------
def save_result(res: SimResult, path) -> None:
    """Counts, first-spike times and the recorded raster in one compressed .npz."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raster = res.raster or {"trial": [], "neuron": [], "t": []}
    np.savez_compressed(
        path, counts=res.counts.astype(np.int32), first_spike=res.first_spike.astype(np.float32),
        t_run=res.t_run, r_trial=np.asarray(raster["trial"], dtype=np.int32),
        r_neuron=np.asarray(raster["neuron"], dtype=np.int32),
        r_t=np.asarray(raster["t"], dtype=np.float32),
        meta_keys=np.array(list(res.meta.keys()), dtype=str),
        meta_vals=np.array([str(v) for v in res.meta.values()], dtype=str),
    )


def load_result(path) -> SimResult:
    with np.load(path, allow_pickle=False) as f:
        raster = {"trial": f["r_trial"], "neuron": f["r_neuron"], "t": f["r_t"].astype(float)}
        meta = dict(zip(f["meta_keys"].tolist(), f["meta_vals"].tolist(), strict=True))
        return SimResult(counts=f["counts"].astype(np.int64),
                         first_spike=f["first_spike"].astype(np.float64),
                         t_run=float(f["t_run"]), raster=raster, meta=meta)


def cached(path, run_fn, force: bool = False) -> SimResult:
    """Load ``path`` if it exists, otherwise call ``run_fn()``, save its result and return it."""
    from pathlib import Path

    path = Path(path)
    if path.exists() and not force:
        return load_result(path)
    res = run_fn()
    save_result(res, path)
    return res
