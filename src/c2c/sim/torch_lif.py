"""Fast simulator: the same LIF network in PyTorch, batched over trials, CPU or GPU.

Why a second simulator?  Brian2 runs one trial at a time through generic generated code. Here the
model is fixed, so it can be written as a few tensor operations on a (neurons x trials) state and
run many trials at once. Every trial is an independent column, so "30 trials" and "a sweep over
8 stimulus rates" are both just a wider batch.

State
-----
v, g     (N, B) membrane potential and synaptic drive in mV, N neurons, B trials
spikes   kept as FLAT indices ``neuron * B + trial`` rather than dense (N, B) booleans. In the fly
         brain under sensory drive only a few hundred of 134k neurons ever fire, so a list of who
         spiked is tiny, while a dense mask would be 4 million entries per step.

One time step, in Brian2's order (see brian_lif.py), so the two simulators agree spike for spike
when given the same input:

    1. groups      v, g <- exact propagator (params.py); refractory neurons stay at v_rst, g = 0
    2. thresholds  spikes = neurons with v > v_th
    3. synapses    g += w_syn * W @ (spikes from D steps ago);  v += 68.75 mV for Poisson events
    4. resets      spiking neurons: v = v_rst, g = 0

Refractoriness without a mask
-----------------------------
In Brian2, variables marked "(unless refractory)" cannot be written at all while a neuron is
refractory: not by integration and not by incoming synapses (input during that window is LOST).
Since the reset sets v = v_rst and g = 0, a refractory neuron therefore sits at exactly
(v_rst, 0) until the period ends. So instead of testing "is this neuron refractory?" for all N x B
entries every step, we remember who spiked in the last R-1 steps (R = refractory steps) and
pin just those entries back to (v_rst, 0) after integration and after synaptic delivery.
Neurons receiving external (Poisson or forced) input have no refractory period, as in the paper,
so their kicks never need masking.

Spike propagation on a sparse W, two ways
-----------------------------------------
* ``mode="event"`` (default): for the few (neuron, trial) pairs that spiked, gather only THEIR
  outgoing synapses from a CSR copy of W transposed (row j = targets of j) and scatter-add the
  weights into g with ``index_add_``. Work ~ spikes x out-degree. This is how Brian2 and most
  spiking simulators deliver spikes, and it is the fast choice on a CPU.
* ``mode="spmm"``: one sparse x dense matrix product per step, g += W @ S, with W stored as a
  ``torch.sparse_csr_tensor`` and S the dense (N, B) spike matrix. Work ~ nnz(W) x B whether or
  not anything fired, but it is a single well-tuned library kernel, so it wins when activity is
  dense or on a GPU with large batches.

Stepping (Phase 4)
------------------
A closed-loop controller cannot hand the brain its whole input up front: the input at 50 ms depends
on what the body did at 49 ms. So all of the simulator's state (v, g, the delay and refractory
rings, the random generator, the spike counters) lives in a ``LIFStepper`` object, and
``stepper.step(n, rate_hz)`` advances it n time steps with the given input rates and returns who
spiked. Calling ``step`` 100 times with 10 steps each is EXACTLY the same computation as one call
with 1000 steps (same operations, same random numbers, same order), which the tests check spike for
spike. ``simulate_torch`` is now just "make a stepper, step it once, collect the result".
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy import sparse

from c2c.sim.network import SimResult, per_trial_rates
from c2c.sim.params import SHIU2024, LIFParams


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch_csr(W: sparse.csr_matrix, dtype, device) -> torch.Tensor:
    """scipy CSR -> torch sparse CSR. Both store the same three arrays (indptr, indices, data)."""
    W = W.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(W.indptr, dtype=torch.int64),
        torch.as_tensor(W.indices, dtype=torch.int64),
        torch.as_tensor(W.data, dtype=dtype),
        size=W.shape, device=device,
    )


class EventPropagator:
    """Event-driven delivery: scatter the outgoing synapses of each spike into g.

    Uses W transposed in CSR form, so row j lists neuron j's TARGETS:
        targets of j = indices[indptr[j] : indptr[j+1]],  weights = data[same slice]
    For a batch of spikes we build the list of all their edges at once, without a Python loop,
    using repeat_interleave (see comments in ``add_``).
    """

    def __init__(self, W: sparse.csr_matrix, scale: float, dtype, device):
        Wt = W.T.tocsr()  # rows = sources
        self.indptr = torch.as_tensor(Wt.indptr, dtype=torch.int64, device=device)
        self.indices = torch.as_tensor(Wt.indices, dtype=torch.int64, device=device)
        self.data = torch.as_tensor(Wt.data * scale, dtype=dtype, device=device)
        self.out_degree = self.indptr[1:] - self.indptr[:-1]

    def add_(self, g: torch.Tensor, flat_spikes: torch.Tensor) -> None:
        """g[target, trial] += weight, for every synapse of every spike in ``flat_spikes``."""
        B = g.shape[1]
        src, trial = flat_spikes // B, flat_spikes % B
        deg = self.out_degree[src]                  # number of synapses leaving each spike
        n_edges = int(deg.sum())
        if n_edges == 0:
            return
        # Example: spikes from neurons with degrees [3, 2] -> 5 edges.
        #   start  = row start of each spike, repeated deg times    [s0 s0 s0 s1 s1]
        #   offset = position within its own row                    [0  1  2  0  1 ]
        start = torch.repeat_interleave(self.indptr[src], deg)
        first_edge_of_spike = torch.repeat_interleave(torch.cumsum(deg, 0) - deg, deg)
        offset = torch.arange(n_edges, device=g.device) - first_edge_of_spike
        edge = start + offset
        target = self.indices[edge]
        b = torch.repeat_interleave(trial, deg)
        # (target, b) of a contiguous (N, B) tensor lives at flat position target * B + b.
        # index_add_ accumulates duplicates, so two spikes onto one target both count.
        g.view(-1).index_add_(0, target * B + b, self.data[edge])


class SpmmPropagator:
    """Delivery as one sparse x dense product per step: g += (w_syn * W) @ S."""

    def __init__(self, W: sparse.csr_matrix, scale: float, dtype, device):
        self.W = to_torch_csr(W * scale, dtype, device)
        self.dtype = dtype

    def add_(self, g: torch.Tensor, flat_spikes: torch.Tensor) -> None:
        S = torch.zeros(g.shape, dtype=self.dtype, device=g.device)
        S.view(-1)[flat_spikes] = 1.0
        g += torch.sparse.mm(self.W, S)


class LIFStepper:
    """The whole simulator state, advanced a few steps at a time.

    Constructor arguments are the ones of ``simulate_torch`` that describe the NETWORK and its
    wiring to the outside world (who gets input, who is recorded). What changes over time, the
    input rates, is passed to ``step``.

        brain = LIFStepper(W, drive=looming_idx, n_trials=1, watch=escape_idx)
        for tick in range(1000):                         # 1000 ticks of 1 ms
            spikes = brain.step(10, rate_hz=encoder(obs)) # (len(watch), n_trials) spike counts
        res = brain.result()                             # the usual SimResult for the whole run

    drive  : neurons that receive Poisson input; their rates are set per ``step`` call
    watch  : neurons whose spike counts ``step`` returns (default: all of them)
    record : neurons whose spike times go into the raster (default: all)
    forced : (neuron indices, times in ms) of single deterministic kicks, as in simulate_torch
    """

    @torch.no_grad()
    def __init__(
        self,
        W: sparse.csr_matrix,
        params: LIFParams = SHIU2024,
        drive=(),
        rate_hz=0.0,
        n_trials: int = 1,
        seed: int = 0,
        forced=None,
        i_bias=0.0,
        record=None,
        record_v=None,
        watch=None,
        mode: str = "event",
        device=None,
        dtype=torch.float32,
    ):
        self.params = params
        self.device = device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        self.mode = mode
        self.gen = torch.Generator(device=device).manual_seed(seed)

        self.n, self.B = n, B = W.shape[0], n_trials
        self.D = params.delay_steps
        self.R = params.refractory_steps
        self.p_vv, self.p_vg, self.p_gg = params.propagators()
        self.propagator = {"event": EventPropagator, "spmm": SpmmPropagator}[mode](
            W, params.w_syn, dtype, device)

        # --- external input ----------------------------------------------------------------
        self.drive_np = np.asarray(drive, dtype=np.int64)
        self.drive_t = torch.as_tensor(self.drive_np, device=device)
        self.set_rates(rate_hz)
        self.forced_at: dict[int, torch.Tensor] = {}
        stimulated = set(self.drive_np.tolist())
        if forced is not None:
            f_idx = np.asarray(forced[0], dtype=np.int64)
            f_step = np.array([params.steps(t) for t in np.asarray(forced[1], dtype=np.float64)])
            for k in np.unique(f_step):
                self.forced_at[int(k)] = torch.as_tensor(f_idx[f_step == k], device=device)
            stimulated |= set(f_idx.tolist())
        # Neurons that get external input have no refractory period (as in the paper).
        self.has_refractory = torch.ones(n, dtype=torch.bool, device=device)
        if stimulated:
            self.has_refractory[sorted(stimulated)] = False

        bias = np.broadcast_to(np.asarray(i_bias, dtype=np.float64), (n,))
        self.bias_term = None
        if np.any(bias != 0):  # constant drive enters like g but never decays: u += (1-P_vv) * I
            self.bias_term = torch.as_tensor(bias * (1.0 - self.p_vv), dtype=dtype,
                                             device=device)[:, None]

        # --- what to report ----------------------------------------------------------------
        self.keep = torch.ones(n, dtype=torch.bool, device=device)
        if record is not None:
            self.keep[:] = False
            self.keep[torch.as_tensor(np.asarray(record, dtype=np.int64), device=device)] = True
        self.vrec = None if record_v is None else torch.as_tensor(np.asarray(record_v),
                                                                  device=device)
        self.watch = None
        if watch is not None:
            w = torch.as_tensor(np.asarray(watch, dtype=np.int64), device=device)
            # flat indices of (watched neuron, trial), in (len(watch), B) order
            self.watch = (w[:, None] * B + torch.arange(B, device=device)[None, :]).reshape(-1)
        self.reset_state()

    def set_rates(self, rate_hz) -> None:
        """Poisson rate (Hz) of every driven neuron: scalar, (n_trials,) or (n_drive, n_trials).
        Stored as the probability of one event per time step, rate * dt."""
        self.p_event = torch.as_tensor(
            per_trial_rates(rate_hz, len(self.drive_np), self.B) * self.params.dt / 1000.0,
            dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def reset_state(self) -> None:
        """Back to t = 0: everyone at rest, no spikes in flight, counters cleared. (The random
        generator is NOT re-seeded, so a reset run is a new, independent sample.)"""
        n, B, dev, dt = self.n, self.B, self.device, self.dtype
        self.k = 0                                                   # steps done so far
        self.v = torch.full((n, B), self.params.v_0, dtype=dt, device=dev)
        self.g = torch.zeros((n, B), dtype=dt, device=dev)
        self.empty = torch.zeros(0, dtype=torch.int64, device=dev)
        self.delay_ring = [self.empty] * max(self.D, 1)             # spikes waiting D steps
        self.refr_ring = [self.empty] * max(self.R - 1, 1)          # spikes of the last R-1 steps
        self.counts = torch.zeros(n * B, dtype=torch.int32, device=dev)
        self.first = torch.full((n * B,), -1, dtype=torch.int32, device=dev)
        self.raster_chunks = []
        self.v_chunks = []
        self.wall_s = 0.0

    @property
    def t_ms(self) -> float:
        """Simulated time so far in ms."""
        return self.k * self.params.dt

    @torch.no_grad()
    def step(self, n_steps: int = 1, rate_hz=None, verbose: bool = False) -> np.ndarray:
        """Advance ``n_steps`` time steps. ``rate_hz`` (if given) replaces the input rates first
        and is held constant over these steps (zero-order hold).

        Returns the number of spikes each watched neuron fired during these steps, as an int array
        of shape (len(watch), n_trials) (all neurons if ``watch`` was None).
        """
        if rate_hz is not None:
            self.set_rates(rate_hz)
        P = self.params
        v, g, B = self.v, self.g, self.B
        D, R = self.D, self.R
        p_vv, p_vg, p_gg = self.p_vv, self.p_vg, self.p_gg
        empty = self.empty
        before = (self.counts if self.watch is None else self.counts[self.watch]).clone()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            k = self.k
            if self.vrec is not None:   # sampled at the START of the step, like Brian2's monitor
                self.v_chunks.append(v[self.vrec].clone())
            refractory = torch.cat(self.refr_ring) if R > 1 else empty

            # 1. groups: exact propagator for everyone, then pin refractory neurons back
            v.sub_(P.v_0).mul_(p_vv).add_(g, alpha=p_vg).add_(P.v_0)
            if self.bias_term is not None:
                v.add_(self.bias_term)
            g.mul_(p_gg)
            if refractory.numel():
                v.view(-1)[refractory] = P.v_rst
                g.view(-1)[refractory] = 0.0

            # 2. thresholds (refractory neurons sit at v_rst < v_th, so they cannot fire)
            spikes = (v.view(-1) > P.v_th).nonzero().squeeze(1)

            # 3. synapses: spikes from step k - D arrive, then this step's external kicks
            if D == 0:
                arriving = spikes
            else:
                arriving = self.delay_ring[k % D]
                self.delay_ring[k % D] = spikes
            if arriving.numel():
                self.propagator.add_(g, arriving)
                if refractory.numel():                 # writes to refractory neurons are lost
                    g.view(-1)[refractory] = 0.0
            if len(self.drive_np):
                events = torch.rand(self.p_event.shape, generator=self.gen, device=self.device,
                                    dtype=self.dtype)
                v[self.drive_t] += P.w_poi * (events < self.p_event).to(self.dtype)
            if k in self.forced_at:
                idx = self.forced_at[k]
                v.index_add_(0, idx, torch.full((len(idx), B), P.w_poi, dtype=self.dtype,
                                                device=self.device))

            # 4. resets, bookkeeping
            if spikes.numel():
                v.view(-1)[spikes] = P.v_rst
                g.view(-1)[spikes] = 0.0
                self.counts[spikes] += 1
                fs = self.first[spikes]
                self.first[spikes] = torch.where(fs < 0, k, fs)
                neuron = spikes // B
                kept = spikes[self.keep[neuron]]
                if kept.numel():
                    self.raster_chunks.append(torch.stack([kept, torch.full_like(kept, k)], 1))
                if R > 1:
                    self.refr_ring[k % (R - 1)] = spikes[self.has_refractory[neuron]]
            elif R > 1:
                self.refr_ring[k % (R - 1)] = empty
            self.k += 1

            if verbose and self.k % max(1, n_steps // 10) == 0:
                print(f"  step {self.k}/{n_steps}  {time.perf_counter() - t0:.1f} s")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.wall_s += time.perf_counter() - t0
        after = self.counts if self.watch is None else self.counts[self.watch]
        return (after - before).view(-1, B).cpu().numpy().astype(np.int64)

    def result(self) -> SimResult:
        """Everything so far as a SimResult (t_run = simulated time), same as simulate_torch."""
        n, B, dt = self.n, self.B, self.params.dt
        first_ms = self.first.view(n, B).cpu().numpy().astype(np.float64) * dt
        first_ms[first_ms < 0] = np.nan
        raster = {"trial": np.zeros(0, int), "neuron": np.zeros(0, int), "t": np.zeros(0)}
        if self.raster_chunks:
            r = torch.cat(self.raster_chunks).cpu().numpy()
            raster = {"trial": r[:, 0] % B, "neuron": r[:, 0] // B, "t": r[:, 1] * dt}
        v_trace = (torch.stack(self.v_chunks).cpu().numpy().astype(np.float64)
                   if self.v_chunks else None)
        return SimResult(counts=self.counts.view(n, B).cpu().numpy().astype(np.int64),
                         first_spike=first_ms, t_run=self.t_ms, raster=raster, v_trace=v_trace,
                         meta={"simulator": "torch", "mode": self.mode,
                               "device": str(self.device),
                               "dtype": str(self.dtype).replace("torch.", ""),
                               "wall_s": self.wall_s})


def simulate_torch(
    W: sparse.csr_matrix,
    params: LIFParams = SHIU2024,
    drive=(),
    rate_hz=0.0,
    t_run: float = 1000.0,
    n_trials: int = 1,
    seed: int = 0,
    forced=None,
    i_bias=0.0,
    record=None,
    record_v=None,
    mode: str = "event",
    device=None,
    dtype=torch.float32,
    verbose: bool = False,
) -> SimResult:
    """Same arguments and result as ``brian_lif.simulate_brian``, plus:

    record : neuron indices whose individual spike times go into ``result.raster``
        (default None = all neurons; pass a few indices for long whole-brain runs)
    mode   : "event" (default) or "spmm", see the module docstring
    device : "cpu", "cuda", ... (default: GPU if available)
    dtype  : torch.float32 (default, fast) or torch.float64 (bit-for-bit comparisons)

    A one-shot run of a ``LIFStepper``: build it, step through the whole trial, collect.
    """
    sim = LIFStepper(W, params, drive=drive, rate_hz=rate_hz, n_trials=n_trials, seed=seed,
                     forced=forced, i_bias=i_bias, record=record, record_v=record_v,
                     watch=np.zeros(0, dtype=np.int64), mode=mode, device=device, dtype=dtype)
    sim.step(params.steps(t_run), verbose=verbose)
    return sim.result()
