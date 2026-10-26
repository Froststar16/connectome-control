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


@torch.no_grad()
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
    """
    device = torch.device(device) if device is not None else default_device()
    gen = torch.Generator(device=device).manual_seed(seed)

    n, B = W.shape[0], n_trials
    n_steps = params.steps(t_run)
    D = params.delay_steps
    R = params.refractory_steps
    p_vv, p_vg, p_gg = params.propagators()
    propagator = {"event": EventPropagator, "spmm": SpmmPropagator}[mode](
        W, params.w_syn, dtype, device)

    # --- external input --------------------------------------------------------------------
    drive_np = np.asarray(drive, dtype=np.int64)
    drive_t = torch.as_tensor(drive_np, device=device)
    p_event = torch.as_tensor(per_trial_rates(rate_hz, len(drive_np), B) * params.dt / 1000.0,
                              dtype=dtype, device=device)   # rate (Hz) * dt (ms) -> probability
    forced_at: dict[int, torch.Tensor] = {}
    stimulated = set(drive_np.tolist())
    if forced is not None:
        f_idx = np.asarray(forced[0], dtype=np.int64)
        f_step = np.array([params.steps(t) for t in np.asarray(forced[1], dtype=np.float64)])
        for k in np.unique(f_step):
            forced_at[int(k)] = torch.as_tensor(f_idx[f_step == k], device=device)
        stimulated |= set(f_idx.tolist())
    # Neurons that get external input have no refractory period (as in the paper).
    has_refractory = torch.ones(n, dtype=torch.bool, device=device)
    if stimulated:
        has_refractory[sorted(stimulated)] = False

    bias = np.broadcast_to(np.asarray(i_bias, dtype=np.float64), (n,))
    bias_term = None
    if np.any(bias != 0):  # constant drive enters like g but never decays: u += (1-P_vv) * I
        bias_term = torch.as_tensor(bias * (1.0 - p_vv), dtype=dtype, device=device)[:, None]

    # --- state -----------------------------------------------------------------------------
    v = torch.full((n, B), params.v_0, dtype=dtype, device=device)
    g = torch.zeros((n, B), dtype=dtype, device=device)
    empty = torch.zeros(0, dtype=torch.int64, device=device)
    delay_ring = [empty] * max(D, 1)        # spikes waiting D steps for delivery
    refr_ring = [empty] * max(R - 1, 1)     # spikes of the last R-1 steps (still refractory)

    counts = torch.zeros(n * B, dtype=torch.int32, device=device)
    first = torch.full((n * B,), -1, dtype=torch.int32, device=device)
    keep = torch.ones(n, dtype=torch.bool, device=device)
    if record is not None:
        keep[:] = False
        keep[torch.as_tensor(np.asarray(record, dtype=np.int64), device=device)] = True
    raster_chunks = []
    vrec = None if record_v is None else torch.as_tensor(np.asarray(record_v), device=device)
    v_chunks = []

    t0 = time.perf_counter()
    for k in range(n_steps):
        if vrec is not None:       # sampled at the START of the step, like Brian2's StateMonitor
            v_chunks.append(v[vrec].clone())
        refractory = torch.cat(refr_ring) if R > 1 else empty

        # 1. groups: exact propagator for everyone, then pin refractory neurons back
        v.sub_(params.v_0).mul_(p_vv).add_(g, alpha=p_vg).add_(params.v_0)
        if bias_term is not None:
            v.add_(bias_term)
        g.mul_(p_gg)
        if refractory.numel():
            v.view(-1)[refractory] = params.v_rst
            g.view(-1)[refractory] = 0.0

        # 2. thresholds (refractory neurons sit at v_rst < v_th, so they cannot fire)
        spikes = (v.view(-1) > params.v_th).nonzero().squeeze(1)

        # 3. synapses: spikes from step k - D arrive, then this step's external kicks
        if D == 0:
            arriving = spikes
        else:
            arriving = delay_ring[k % D]
            delay_ring[k % D] = spikes
        if arriving.numel():
            propagator.add_(g, arriving)
            if refractory.numel():                 # writes to refractory neurons are lost
                g.view(-1)[refractory] = 0.0
        if len(drive_np):
            events = torch.rand(p_event.shape, generator=gen, device=device, dtype=dtype)
            v[drive_t] += params.w_poi * (events < p_event).to(dtype)
        if k in forced_at:
            v.index_add_(0, forced_at[k],
                         torch.full((len(forced_at[k]), B), params.w_poi, dtype=dtype,
                                    device=device))

        # 4. resets, bookkeeping
        if spikes.numel():
            v.view(-1)[spikes] = params.v_rst
            g.view(-1)[spikes] = 0.0
            counts[spikes] += 1
            fs = first[spikes]
            first[spikes] = torch.where(fs < 0, k, fs)
            neuron = spikes // B
            kept = spikes[keep[neuron]]
            if kept.numel():
                raster_chunks.append(torch.stack([kept, torch.full_like(kept, k)], dim=1))
            if R > 1:
                refr_ring[k % (R - 1)] = spikes[has_refractory[neuron]]
        elif R > 1:
            refr_ring[k % (R - 1)] = empty

        if verbose and (k + 1) % max(1, n_steps // 10) == 0:
            print(f"  step {k + 1}/{n_steps}  {time.perf_counter() - t0:.1f} s")

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    first_ms = first.view(n, B).cpu().numpy().astype(np.float64) * params.dt
    first_ms[first_ms < 0] = np.nan
    raster = {"trial": np.zeros(0, int), "neuron": np.zeros(0, int), "t": np.zeros(0)}
    if raster_chunks:
        r = torch.cat(raster_chunks).cpu().numpy()
        raster = {"trial": r[:, 0] % B, "neuron": r[:, 0] // B, "t": r[:, 1] * params.dt}
    v_trace = torch.stack(v_chunks).cpu().numpy().astype(np.float64) if v_chunks else None
    return SimResult(counts=counts.view(n, B).cpu().numpy().astype(np.int64),
                     first_spike=first_ms, t_run=t_run, raster=raster, v_trace=v_trace,
                     meta={"simulator": "torch", "mode": mode, "device": str(device),
                           "dtype": str(dtype).replace("torch.", ""), "wall_s": wall})
