"""Tests for the spiking simulators on tiny networks whose behaviour can be worked out by hand.

Every test runs on BOTH simulators where it makes sense, because the point of Phase 3 is that
they implement the same model. Numbers used below (Shiu et al. defaults):

    rest = reset = -52 mV, threshold -45 mV  ->  theta = 7 mV gap
    t_mbr = 20 ms, tau = 5 ms, refractory 2.2 ms (22 steps), delay 1.8 ms (18 steps), dt 0.1 ms
    w_syn = 0.275 mV per synapse, Poisson kick 68.75 mV

A single input spike carrying c synapses produces the postsynaptic potential (PSP)

    u(t) = c * w_syn * tau / (tau - t_mbr) * (exp(-t/tau) - exp(-t/t_mbr))

which peaks at t* = ln(t_mbr/tau) * tau * t_mbr / (t_mbr - tau) = 9.24 ms with height
0.1575 * c * w_syn. So one presynaptic spike needs c > 7 / (0.1575 * 0.275) = 162 synapses to
make the target fire on its own; real fly neurons need many inputs together.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, sparse

torch = pytest.importorskip("torch")      # the simulators need the [sim] extras
pytest.importorskip("brian2")

from c2c.data.graph import build_graph  # imported after the skips on purpose
from c2c.data.loaders import POST, PRE
from c2c.sim.brian_lif import simulate_brian
from c2c.sim.network import silence_columns, subnetwork, weight_matrix
from c2c.sim.params import SHIU2024 as P
from c2c.sim.torch_lif import simulate_torch

D, R = P.delay_steps, P.refractory_steps
THETA = P.v_th - P.v_0


def run(sim, W, **kw):
    """Call either simulator with test-friendly defaults (float64, CPU, numpy codegen)."""
    if sim == "torch":
        return simulate_torch(W, P, device="cpu", dtype=torch.float64, **kw)
    return simulate_brian(W, P, codegen="numpy", **kw)


SIMS = ["torch", "brian"]


def chain(c: float, n: int = 2) -> sparse.csr_matrix:
    """0 -> 1 (-> 2 ...) with c synapses per link. Rows are targets: W[1, 0] = c."""
    W = sparse.lil_matrix((n, n))
    for i in range(1, n):
        W[i, i - 1] = c
    return W.tocsr()


def psp(t_ms, c):
    """Analytic PSP (mV above rest) t_ms after a spike with c synapses reaches the target."""
    t = np.asarray(t_ms, dtype=float)
    return c * P.w_syn * P.tau / (P.tau - P.t_mbr) * (np.exp(-t / P.tau) - np.exp(-t / P.t_mbr))


# ---- parameters ------------------------------------------------------------------------
def test_shiu_parameters():
    assert (P.v_0, P.v_rst, P.v_th) == (-52.0, -52.0, -45.0)
    assert (P.t_mbr, P.tau, P.t_rfc, P.t_dly) == (20.0, 5.0, 2.2, 1.8)
    assert P.w_poi == pytest.approx(68.75)
    assert (D, R) == (18, 22)  # rounding, not truncation: 1.8 / 0.1 = 17.999999999999996


def test_propagator_is_the_matrix_exponential():
    """The closed-form step must equal expm(A dt) for d[u, g]/dt = A [u, g]."""
    A = np.array([[-1 / P.t_mbr, 1 / P.t_mbr],
                  [0.0, -1 / P.tau]])
    E = linalg.expm(A * P.dt)
    p_vv, p_vg, p_gg = P.propagators()
    assert np.allclose(E, [[p_vv, p_vg], [0.0, p_gg]], rtol=1e-12, atol=1e-15)


# ---- single neuron ---------------------------------------------------------------------
def discrete_isi(I):
    """Steps between spikes for constant drive I (mV). After a spike at step s the neuron is
    frozen at steps s+1 ... s+R-1 (R-1 steps), then each integration step gives
    u_m = I * (1 - P_vv**m), and it fires at the first m with u_m > theta."""
    p_vv = P.propagators()[0]
    m = math.floor(math.log(1 - THETA / I) / math.log(p_vv)) + 1
    return R - 1 + m


@pytest.mark.parametrize("sim", SIMS)
def test_f_I_curve(sim):
    """Constant drive I: no spikes below the 7 mV gap, then the LIF f-I curve
    f = 1 / (t_rfc + t_mbr * ln(I / (I - theta))), with spike times on the dt grid."""
    currents = np.array([5.0, 6.9, 7.5, 10.0, 20.0, 40.0])
    W = sparse.csr_matrix((len(currents), len(currents)))
    res = run(sim, W, i_bias=currents, t_run=300.0)
    for k, I in enumerate(currents):
        t = np.sort(res.raster["t"][res.raster["neuron"] == k])
        if I <= THETA:
            assert len(t) == 0
            continue
        isi_steps = np.round(np.diff(t) / P.dt).astype(int)
        assert set(isi_steps.tolist()) == {discrete_isi(I)}        # exact on the grid
        f_analytic = 1000.0 / (P.t_rfc + P.t_mbr * math.log(I / (I - THETA)))
        f_sim = 1000.0 / (isi_steps[0] * P.dt)
        # On the grid the neuron is frozen for R-1 = 21 steps (the step that integrates out of
        # the refractory period also counts as time passing), and the threshold crossing is only
        # seen at the end of a step. Those two effects are each < 1 step, in opposite directions.
        assert abs(1000.0 / f_sim - 1000.0 / f_analytic) < P.dt


@pytest.mark.parametrize("sim", SIMS)
def test_poisson_driven_neuron_fires_at_drive_rate(sim):
    """A Poisson kick (68.75 mV) always forces a spike, and driven neurons have no refractory
    period, so the neuron fires as a Poisson process at the drive rate."""
    W = sparse.csr_matrix((1, 1))
    n_trials = 4 if sim == "brian" else 40
    res = run(sim, W, drive=[0], rate_hz=100.0, t_run=1000.0, n_trials=n_trials, seed=3)
    rate = res.rates()[0]
    sem = np.sqrt(100.0 / n_trials)  # Poisson: variance of a 1 s count = rate
    assert abs(rate.mean() - 100.0) < 4 * sem


# ---- two neurons -----------------------------------------------------------------------
@pytest.mark.parametrize("sim", SIMS)
def test_psp_shape_and_synaptic_delay(sim):
    """Kick neuron 0 at t = 5 ms. It spikes one step later (step 51), its spike reaches neuron 1
    D = 18 steps after that (step 69, synapses slot), g starts acting at the next integration
    (step 70), and v is sampled at the start of each step, so the first change shows at step 71.
    From there on v_1 must follow the analytic PSP exactly."""
    c = 20.0
    res = run(sim, chain(c), forced=([0], [5.0]), t_run=40.0, record_v=[1])
    v1 = res.v_trace[:, 0, 0]
    spike_step = 51
    assert res.raster["t"][res.raster["neuron"] == 0].tolist() == pytest.approx([5.1])
    arrival = spike_step + D
    assert np.all(v1[: arrival + 2] == P.v_0)          # nothing before the delay is up
    k = np.arange(arrival + 2, len(v1))
    expected = P.v_0 + psp((k - arrival - 1) * P.dt, c)
    assert np.allclose(v1[k], expected, atol=1e-9)
    assert (v1.max() - P.v_0) == pytest.approx(0.1575 * c * P.w_syn, rel=2e-3)


@pytest.mark.parametrize("sim", SIMS)
def test_strong_excitation_relays_every_spike(sim):
    """300 synapses -> PSP peak 13 mV > 7 mV gap: neuron 1 copies neuron 0, a fixed lag later."""
    times = [5.0, 30.0, 55.0, 80.0]
    res = run(sim, chain(300.0), forced=([0] * 4, times), t_run=100.0)
    t0 = np.sort(res.raster["t"][res.raster["neuron"] == 0])
    t1 = np.sort(res.raster["t"][res.raster["neuron"] == 1])
    assert len(t0) == len(t1) == 4
    lag = t1 - t0
    assert np.allclose(lag, lag[0]) and lag[0] > P.t_dly   # delay + rise time to threshold


@pytest.mark.parametrize("sim", SIMS)
def test_inhibition_slows_and_excitation_does_not(sim):
    """Neuron 1 fires regularly under constant drive. Spikes from neuron 0 through 40 inhibitory
    synapses delay it; the same spikes through 40 excitatory synapses speed it up."""
    kicks = ([0] * 10, list(np.arange(10.0, 110.0, 10.0)))
    rates = {}
    for label, c in [("inh", -40.0), ("none", 0.0), ("exc", 40.0)]:
        W = chain(c) if c else sparse.csr_matrix((2, 2))
        res = run(sim, W, forced=kicks, i_bias=[0.0, 9.0], t_run=120.0)
        rates[label] = res.counts[1, 0]
    assert rates["inh"] < rates["none"] < rates["exc"]


@pytest.mark.parametrize("sim", SIMS)
@pytest.mark.parametrize("steps_after_spike", [5, R - 1])
def test_input_during_refractory_period_is_lost(sim, steps_after_spike):
    """Brian2 blocks every write to '(unless refractory)' variables while a neuron is refractory,
    synaptic writes included, so input arriving in that window vanishes instead of being stored.

    Neuron 1 fires under constant drive (it is not a Poisson/forced target, so it HAS a
    refractory period). We time a spike of neuron 0 to land k steps after neuron 1's first spike
    and check that neuron 1's spike train is unchanged. k = R-1 is the last refractory step, the
    one edge case where a stored-up input would leak into the next integration step."""
    drive = [0.0, 30.0]
    base = run(sim, sparse.csr_matrix((2, 2)), i_bias=drive, t_run=30.0)
    t1 = np.sort(base.raster["t"][base.raster["neuron"] == 1])
    # kick at t -> neuron 0 spikes at t + dt -> arrives at t + dt + t_dly
    kick_t = round(t1[0] - P.dt - P.t_dly + steps_after_spike * P.dt, 1)
    res = run(sim, chain(20.0), forced=([0], [kick_t]), i_bias=drive, t_run=30.0)
    assert np.array_equal(t1, np.sort(res.raster["t"][res.raster["neuron"] == 1]))


# ---- Brian2 vs PyTorch ------------------------------------------------------------------
def random_network(n=60, density=0.15, seed=1):
    rng = np.random.default_rng(seed)
    W = sparse.random(n, n, density=density, random_state=seed, format="csr")
    W.data = rng.integers(5, 40, W.nnz) * np.where(rng.random(W.nnz) < 0.7, 1.0, -1.0)
    W.setdiag(0)
    W.eliminate_zeros()
    return W


def test_brian_and_torch_agree_spike_for_spike_with_identical_input():
    """Same network, same deterministic input -> the same spikes at the same time steps, and the
    same membrane potentials to ~1e-12 mV (float64). Both event and spmm delivery."""
    W = random_network()
    rng = np.random.default_rng(7)
    idx = rng.integers(0, 10, 300)
    t = np.round(rng.uniform(0, 200, 300), 1)
    pairs = np.unique(np.stack([idx, np.round(t * 10)]), axis=1)  # no duplicate (neuron, step)
    forced = (pairs[0].astype(int), pairs[1] / 10)
    rb = run("brian", W, forced=forced, t_run=200.0, record_v=[12, 30])
    assert rb.counts.sum() > 300            # the recurrent network is actually doing something
    for mode in ["event", "spmm"]:
        rt = run("torch", W, forced=forced, t_run=200.0, record_v=[12, 30], mode=mode)
        sb = sorted(zip(rb.raster["neuron"], np.round(rb.raster["t"], 6), strict=True))
        st = sorted(zip(rt.raster["neuron"], np.round(rt.raster["t"], 6), strict=True))
        assert sb == st
        assert np.abs(rb.v_trace - rt.v_trace).max() < 1e-9


def test_brian_and_torch_agree_statistically_with_poisson_input():
    """With Poisson input the two simulators use different random numbers, so only the
    statistics can match: per-neuron mean rates within sampling error."""
    W = random_network(n=80, density=0.12, seed=4)
    drive = np.arange(8)
    kw = {"drive": drive, "rate_hz": 80.0, "t_run": 500.0}
    rb = run("brian", W, n_trials=12, seed=10, **kw)
    rt = run("torch", W, n_trials=200, seed=11, **kw)
    mb, mt = rb.mean_rate(), rt.mean_rate()
    se = np.sqrt(rb.sem_rate() ** 2 + rt.sem_rate() ** 2) + 0.5  # +0.5 Hz floor for silent cells
    assert mb.sum() > 0
    assert np.all(np.abs(mb - mt) < 5 * se)
    assert np.corrcoef(mb, mt)[0, 1] > 0.98


def test_event_and_spmm_and_float32_agree():
    W = random_network(seed=2)
    kw = {"drive": np.arange(6), "rate_hz": 150.0, "t_run": 200.0, "n_trials": 8, "seed": 5}
    a = simulate_torch(W, P, mode="event", device="cpu", dtype=torch.float64, **kw)
    b = simulate_torch(W, P, mode="spmm", device="cpu", dtype=torch.float64, **kw)
    assert np.array_equal(a.counts, b.counts)
    c = simulate_torch(W, P, mode="event", device="cpu", dtype=torch.float32, **kw)
    # float32 rounding can flip a near-threshold spike, so statistics only
    assert abs(c.counts.sum() - a.counts.sum()) <= 0.05 * a.counts.sum()


def test_per_trial_rates_make_a_sweep_in_one_batch():
    W = sparse.csr_matrix((1, 1))
    rates = np.repeat([0.0, 50.0, 200.0], 20)
    res = simulate_torch(W, P, drive=[0], rate_hz=rates, t_run=500.0, n_trials=len(rates),
                         device="cpu", seed=1)
    m = res.rates()[0].reshape(3, 20).mean(axis=1)
    assert m[0] == 0 and 30 < m[1] < 70 and 170 < m[2] < 230


# ---- weight matrix helpers ---------------------------------------------------------------
@pytest.fixture
def g():
    edges = {
        PRE: [100, 100, 200, 300],
        POST: [200, 300, 300, 100],
        "syn_count": [10, 6, 8, 7],
        "nt_type": ["ACH", "SER", "GABA", "GLUT"],
    }
    return build_graph(pd.DataFrame(edges))


def test_weight_matrix_modulatory_sign(g):
    W0 = weight_matrix(g).toarray()
    W1 = weight_matrix(g, modulatory_sign=1).toarray()
    # indices: 100 -> 0, 200 -> 1, 300 -> 2; rows are targets
    assert W0[1, 0] == 10 and W0[2, 1] == -8 and W0[0, 2] == -7
    assert W0[2, 0] == 0 and W1[2, 0] == 6        # the serotonergic edge


def test_silencing_removes_outgoing_edges_only(g):
    W = weight_matrix(g)
    Ws = silence_columns(W, [0]).toarray()        # mute neuron 0 (root id 100)
    assert np.all(Ws[:, 0] == 0)                  # nothing leaves neuron 0
    assert Ws[0, 2] == -7                         # but its input is untouched


def test_subnetwork_keeps_internal_edges(g):
    sub, keep = subnetwork(weight_matrix(g), [2, 0])
    assert keep.tolist() == [0, 2]
    assert sub.toarray().tolist() == [[0, -7], [0, 0]]
