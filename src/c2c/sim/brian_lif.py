"""Reference simulator: the Shiu et al. (2024) whole-brain LIF model in Brian2.

Brian2 (brian2.readthedocs.io) takes the model as equations written out in text with physical
units, generates code for them, and runs it. That makes it slow-ish but very hard to get wrong,
which is exactly what a reference implementation is for. The equations, threshold, reset and
refractoriness below are the authors' own (model.py in their repository), with three deliberate
changes, none of which alters the dynamics:

1. Their reset string is 'v = v_rst; w = 0; g = 0 * mV'. ``w`` is not a neuron variable (it is
   the SYNAPSE weight), so the assignment only sets a throwaway local; we drop it for clarity.
2. Poisson drive uses a PoissonGroup + Synapses instead of one PoissonInput per neuron. Both draw
   an independent Bernoulli(rate * dt) event per neuron per time step and add 68.75 mV to v in
   the same scheduling slot, so the statistics are identical; the group form lets us change the
   rate between trials without rebuilding the 2.6-million-synapse network.
3. Optional extras for testing only: a constant bias drive ``i_bias`` (default 0, needed for the
   analytic f-I curve) and deterministic "forced" kicks (a SpikeGeneratorGroup), so that the
   PyTorch simulator can be compared with this one spike by spike.

Brian2's time step, in order (its default schedule): integrate the ODEs ("groups"), check the
threshold ("thresholds"), deliver synaptic events ("synapses"), reset ("resets"). The PyTorch
simulator reproduces this order exactly; it matters for edge cases such as an input arriving in
the same step as a spike (the reset wipes it out).
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

from c2c.sim.network import SimResult, edge_list, per_trial_rates
from c2c.sim.params import SHIU2024, LIFParams

EQS = """
dv/dt = (v_0 - v + g + i_bias) / t_mbr : volt (unless refractory)
dg/dt = -g / tau                         : volt (unless refractory)
rfc    : second
i_bias : volt
"""


def simulate_brian(
    W: sparse.csr_matrix,
    params: LIFParams = SHIU2024,
    drive=(),
    rate_hz=0.0,
    t_run: float = 1000.0,
    n_trials: int = 1,
    seed: int = 0,
    forced=None,
    i_bias=0.0,
    record_v=None,
    codegen: str = "auto",
    verbose: bool = False,
) -> SimResult:
    """Run ``n_trials`` independent trials and return spike counts and first-spike times.

    Parameters
    ----------
    W : signed synapse-count matrix, rows = targets (see network.weight_matrix)
    drive : indices of neurons that receive Poisson input (their refractory period is set to 0,
        as in the paper)
    rate_hz : Poisson rate; scalar, one value per trial, or (len(drive), n_trials)
    t_run : trial length in ms
    forced : optional (neuron_indices, times_ms) of deterministic 68.75 mV kicks (testing)
    i_bias : constant drive in mV added inside the membrane equation (testing; default 0)
    record_v : optional indices whose membrane potential is recorded every step
    codegen : Brian2 code-generation target: "auto", "cython" or "numpy"
    """
    import brian2 as b2  # imported here so the rest of the package works without Brian2
    from brian2 import Hz, ms, mV

    b2.prefs.codegen.target = codegen
    b2.start_scope()
    b2.defaultclock.dt = params.dt * ms

    n = W.shape[0]
    drive = np.asarray(drive, dtype=np.int64)
    rate_table = per_trial_rates(rate_hz, len(drive), n_trials)  # not "rates": Brian2 would
    # confuse a local variable called "rates" with the PoissonGroup variable of the same name

    namespace = {
        "v_0": params.v_0 * mV, "v_rst": params.v_rst * mV, "v_th": params.v_th * mV,
        "t_mbr": params.t_mbr * ms, "tau": params.tau * ms, "w_poi": params.w_poi * mV,
    }

    neu = b2.NeuronGroup(
        n, EQS, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0*mV",
        refractory="rfc", namespace=namespace, name="neurons",
    )
    neu.v = params.v_0 * mV
    neu.g = 0 * mV
    neu.rfc = params.t_rfc * ms
    neu.i_bias = np.broadcast_to(np.asarray(i_bias, dtype=float), (n,)) * mV

    objects = [neu]

    # Recurrent synapses: one Brian2 synapse per stored edge, i = presynaptic, j = postsynaptic.
    pre, post, w = edge_list(W)
    if len(pre):
        syn = b2.Synapses(neu, neu, "w : volt", on_pre="g_post += w",
                          delay=params.t_dly * ms, name="recurrent")
        syn.connect(i=pre, j=post)
        syn.w = w * params.w_syn * mV
        objects.append(syn)

    stimulated = set(drive.tolist())

    if len(drive):
        pg = b2.PoissonGroup(len(drive), rates=rate_table[:, 0] * Hz, name="poisson")
        pg_syn = b2.Synapses(pg, neu, on_pre="v_post += w_poi", namespace=namespace,
                             name="poisson_syn")
        pg_syn.connect(i=np.arange(len(drive)), j=drive)
        objects += [pg, pg_syn]

    if forced is not None:
        f_idx = np.asarray(forced[0], dtype=np.int64)
        f_t = np.asarray(forced[1], dtype=np.float64)
        f_targets, gen_idx = np.unique(f_idx, return_inverse=True)
        gen = b2.SpikeGeneratorGroup(len(f_targets), gen_idx, f_t * ms, name="forced")
        gen_syn = b2.Synapses(gen, neu, on_pre="v_post += w_poi", namespace=namespace,
                              name="forced_syn")
        gen_syn.connect(i=np.arange(len(f_targets)), j=f_targets)
        objects += [gen, gen_syn]
        stimulated |= set(f_targets.tolist())

    if stimulated:
        neu.rfc[np.array(sorted(stimulated))] = 0 * ms  # Poisson targets: no refractory period

    spk = b2.SpikeMonitor(neu, name="spikes")
    objects.append(spk)
    vmon = None
    if record_v is not None:
        vmon = b2.StateMonitor(neu, "v", record=np.asarray(record_v), name="vmon")
        objects.append(vmon)

    net = b2.Network(*objects)
    net.store()  # snapshot of the initial state; every trial restarts from here

    counts = np.zeros((n, n_trials), dtype=np.int64)
    first = np.full((n, n_trials), np.nan)
    raster = {"trial": [], "neuron": [], "t": []}
    v_trace = [] if vmon is not None else None
    t0 = time.perf_counter()
    for trial in range(n_trials):
        net.restore()
        b2.seed(seed + trial)
        if len(drive):
            pg.rates = rate_table[:, trial] * Hz
        net.run(t_run * ms, namespace={})  # {}: never pick up names from this function
        i = np.asarray(spk.i[:], dtype=np.int64)
        t = np.asarray(spk.t / ms)
        counts[:, trial] = np.bincount(i, minlength=n)
        # first spike: spikes are stored in time order, so the first occurrence of each neuron
        uniq, pos = np.unique(i, return_index=True)
        first[uniq, trial] = t[pos]
        raster["trial"].append(np.full(len(i), trial))
        raster["neuron"].append(i)
        raster["t"].append(t)
        if vmon is not None:
            v_trace.append(np.asarray(vmon.v / mV).T)  # (n_steps, n_recorded)
        if verbose:
            print(f"  brian2 trial {trial + 1}/{n_trials}: {len(i):,} spikes, "
                  f"{time.perf_counter() - t0:.1f} s elapsed")

    raster = {k: np.concatenate(v) for k, v in raster.items()}
    if v_trace is not None:
        v_trace = np.stack(v_trace, axis=-1)
    return SimResult(counts=counts, first_spike=first, t_run=t_run, raster=raster,
                     v_trace=v_trace,
                     meta={"simulator": "brian2", "wall_s": time.perf_counter() - t0,
                           "codegen": codegen})
