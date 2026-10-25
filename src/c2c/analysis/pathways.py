"""Tracing signals through the connectome: hop layers, strongest paths, subgraphs, signed effects
and a probabilistic "information flow" layering.

Every function takes a ``Graph`` (Phase 1) and works on matrix indices, never root IDs; use
``g.index_of`` / ``annotations.select`` to get indices. Conventions inherited from Phase 1:

    counts[i, j] = synapses FROM j (source, presynaptic) ONTO i (target, postsynaptic)
    sign[e]      = +1 / -1 / 0 for the e-th stored edge, aligned with counts.data

That orientation means "who receives input from the set f" is one sparse matrix-vector product,
``counts @ f``, which is the workhorse of this module.

Weights: a raw synapse count is a poor measure of influence, because a neuron with 5,000 inputs
cares less about any one 20-synapse connection than a neuron with 100 inputs does. So, like the
FlyWire papers, we use the INPUT FRACTION

    w[i, j] = counts[i, j] / (total input synapses of i)

i.e. the share of i's input that comes from j. Each row of w sums to 1. Note the denominator is
the total in the thresholded (>=5 synapse) graph, which slightly overstates fractions compared
with counting every weak synapse; pass ``total_input`` to override.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from c2c.data.graph import Graph

UNREACHED = -1          # hop distance of a neuron no path reaches
NO_PREDECESSOR = -9999  # what scipy.sparse.csgraph uses for "no previous node"
MIN_COST = 1e-12        # keeps -log(1) = 0 edges strictly positive (see strongest_paths)


# ========================================================================================
# Small helpers
# ========================================================================================
def _as_mask(n: int, nodes) -> np.ndarray:
    """Index list -> boolean vector of length n (a set, in vector form)."""
    nodes = np.asarray(nodes, dtype=np.int64).ravel()
    if nodes.size == 0:
        raise ValueError("seed set is empty")
    mask = np.zeros(n, dtype=bool)
    mask[nodes] = True
    return mask


def edge_rows(m: sparse.csr_matrix) -> np.ndarray:
    """Row index of every stored entry, in storage order (the partner of ``m.indices``)."""
    return np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))


def input_fractions(g: Graph, total_input=None) -> sparse.csr_matrix:
    """Same sparsity pattern AND storage order as ``g.counts``, with data = input fraction.

    We divide ``data`` in place instead of computing ``diag(1/total) @ counts``, because a
    sparse matrix product is free to reorder the stored entries, which would break the
    alignment with ``g.sign``. Copy-and-edit-data keeps edge e at position e.
    """
    F = g.counts.astype(np.float64)  # astype copies data; indices/indptr keep their order
    if total_input is None:
        total = np.asarray(g.counts.sum(axis=1, dtype=np.int64)).ravel()
    else:
        total = np.asarray(total_input, dtype=np.float64)
    F.data /= total[edge_rows(F)]
    return F


# ========================================================================================
# 1. Hop distance (breadth-first search)
# ========================================================================================
def hop_distance(g: Graph, seeds, direction: str = "downstream",
                 max_hops: int | None = None) -> np.ndarray:
    """Fewest synaptic steps from ANY seed to every neuron (multi-source BFS).

    BFS as linear algebra: keep the current frontier as a boolean vector f. A neuron i is in the
    next layer if it receives at least one edge from the frontier, i.e. (counts @ f)[i] > 0, and
    has not been visited yet. Each layer costs one pass over the edges, O(E), and the brain is
    only ~7 layers deep, so this runs in a fraction of a second on 2.7M edges.

    direction="upstream" walks edges backwards (who can reach the seeds?) using counts.T.
    Returns int32 array: 0 for seeds, k for k hops, UNREACHED (-1) if no path exists.
    """
    if direction == "downstream":
        A = g.counts
    elif direction == "upstream":
        A = g.counts.T.tocsr()
    else:
        raise ValueError("direction must be 'downstream' or 'upstream'")

    dist = np.full(g.n, UNREACHED, dtype=np.int32)
    frontier = _as_mask(g.n, seeds)
    dist[frontier] = 0
    hop = 0
    while frontier.any() and (max_hops is None or hop < max_hops):
        hop += 1
        touched = (A @ frontier.astype(np.int32)) > 0  # has an edge from the frontier
        frontier = touched & (dist == UNREACHED)       # ...and was not reached earlier
        dist[frontier] = hop
    return dist


def hop_layers(dist: np.ndarray) -> dict[int, np.ndarray]:
    """{hop: indices of neurons at exactly that many hops}, unreached neurons left out."""
    return {h: np.flatnonzero(dist == h) for h in range(int(dist.max()) + 1)}


# ========================================================================================
# 2. Strongest paths (Dijkstra on -log input fraction)
# ========================================================================================
@dataclass
class StrongestPaths:
    """Result of a single-pass Dijkstra from a seed set to every neuron.

    cost[i]        -log of the best product of input fractions along any path seed -> i
    predecessor[i] previous neuron on that best path (NO_PREDECESSOR for seeds / unreached)
    source[i]      the seed that best path starts from
    """

    cost: np.ndarray
    predecessor: np.ndarray
    source: np.ndarray

    @property
    def strength(self) -> np.ndarray:
        """exp(-cost) = product of input fractions along the best path (0 if unreachable)."""
        return np.exp(-self.cost)

    def path_to(self, target: int) -> list[int]:
        """Follow predecessors back from ``target``; returned in signal order (seed first)."""
        if not np.isfinite(self.cost[target]):
            return []
        path = [int(target)]
        while self.predecessor[path[-1]] != NO_PREDECESSOR:
            path.append(int(self.predecessor[path[-1]]))
        return path[::-1]


def strongest_paths(g: Graph, sources, fractions: sparse.csr_matrix | None = None
                    ) -> StrongestPaths:
    """Path that maximises the product of input fractions, from the seed set to every neuron.

    Why -log: Dijkstra minimises a SUM of non-negative edge costs. Maximising a product of
    fractions w1*w2*...*wk is the same as minimising -log(w1) - log(w2) - ... - log(wk), and
    each -log(w) >= 0 because 0 < w <= 1. A strong edge (w near 1) costs ~0, a weak one is
    expensive, so the shortest path in cost is the strongest chain in the circuit.

    Why one call: ``min_only=True`` treats all seeds as a single super-source, so we get the best
    path from *any* seed to every neuron in one O(E log V) pass instead of one pass per seed.

    Edge case: a neuron whose only input is one partner has w = 1 and cost exactly 0. We clamp
    costs at MIN_COST so a zero never gets confused with "no edge".
    """
    F = input_fractions(g) if fractions is None else fractions
    cost = F.copy()
    cost.data = np.maximum(-np.log(cost.data), MIN_COST)
    # csgraph reads graph[u, v] as an edge u -> v; ours stores j -> i at [i, j], so transpose.
    dist, pred, src = csgraph.dijkstra(
        cost.T.tocsr(), directed=True, indices=np.asarray(sources, dtype=np.int64),
        return_predecessors=True, min_only=True,
    )
    return StrongestPaths(cost=dist, predecessor=pred, source=src)


# ========================================================================================
# 3. Signed path effect
# ========================================================================================
def edge_position(g: Graph, pre: int, post: int) -> int:
    """Position of edge pre -> post in ``counts.data`` / ``sign`` (KeyError if absent).

    Row ``post`` holds all inputs of ``post``; its column indices are sorted (we built the CSR
    that way), so binary search finds ``pre`` in O(log degree).
    """
    start, end = g.counts.indptr[post], g.counts.indptr[post + 1]
    k = start + np.searchsorted(g.counts.indices[start:end], pre)
    if k >= end or g.counts.indices[k] != pre:
        raise KeyError(f"no edge {pre} -> {post}")
    return int(k)


@dataclass
class PathEffect:
    nodes: list[int]
    synapses: np.ndarray   # per edge
    fraction: np.ndarray   # per edge, input fraction
    sign: np.ndarray       # per edge, +1 / -1 / 0

    @property
    def strength(self) -> float:
        return float(np.prod(self.fraction))

    @property
    def net_sign(self) -> int:
        """Product of signs: two inhibitions in a row = disinhibition = net excitation (+1).
        Any modulatory (0) step zeroes the chain: it is not a fast excitatory/inhibitory link."""
        return int(np.prod(self.sign))

    @property
    def effect(self) -> float:
        return self.net_sign * self.strength


def path_effect(g: Graph, nodes, fractions: sparse.csr_matrix | None = None) -> PathEffect:
    """Per-edge synapses, fractions and signs along a path given in signal order."""
    F = input_fractions(g) if fractions is None else fractions
    nodes = [int(v) for v in nodes]
    if len(nodes) < 2:
        raise ValueError("a path needs at least two neurons")
    pos = np.array([edge_position(g, a, b) for a, b in pairwise(nodes)])
    return PathEffect(nodes=nodes, synapses=g.counts.data[pos], fraction=F.data[pos],
                      sign=g.sign[pos].astype(np.int64))


def multihop_effect(g: Graph, sources, n_hops: int, signed: bool = True,
                    fractions: sparse.csr_matrix | None = None) -> np.ndarray:
    """Summed path effect of the seed set on every neuron over ALL walks of exactly k hops.

    Let S be the (signed) input-fraction matrix. Starting from x0 = 1 on the seeds,
    x_k = S @ x_{k-1}, so x_k[i] = sum over every k-step walk seed -> i of the product of signed
    fractions along it. That is the "path effect" above, added up over every path at once.
    It is a linear-system view (no thresholds, no spiking), useful for ranking targets, not
    for predicting firing rates. Returns an array of shape (n_hops + 1, n); row 0 is x0.
    """
    S = (input_fractions(g) if fractions is None else fractions).copy()
    if signed:
        S.data = S.data * g.sign  # safe because S keeps the storage order of counts
    x = _as_mask(g.n, sources).astype(np.float64)
    out = [x]
    for _ in range(n_hops):
        x = S @ x
        out.append(x)
    return np.vstack(out)


# ========================================================================================
# 4. Subgraphs and group-level wiring
# ========================================================================================
def subgraph(g: Graph, nodes) -> Graph:
    """The Graph induced by ``nodes``: those neurons and every edge between them.

    The trap: slicing ``counts`` gives the right counts but loses which position each edge had,
    so ``sign`` would no longer line up. The fix is to slice a matrix whose data is the EDGE ID
    (1..nnz; 1-based so no stored value is 0, which scipy may drop), then use the surviving IDs
    to pick counts and signs from the originals.
    """
    idx = np.unique(np.asarray(nodes, dtype=np.int64))
    c = g.counts
    edge_id = sparse.csr_matrix(
        (np.arange(1, c.nnz + 1, dtype=np.int64), c.indices, c.indptr), shape=c.shape
    )
    sub = edge_id[idx][:, idx].tocsr()
    sub.sort_indices()
    ids = sub.data - 1
    counts = sparse.csr_matrix((c.data[ids], sub.indices, sub.indptr), shape=sub.shape)
    return Graph(root_ids=g.root_ids[idx], counts=counts, sign=g.sign[ids],
                 home_neuropil=g.home_neuropil[idx])


def edge_table(g: Graph) -> pd.DataFrame:
    """Long-form edge list: one row per stored edge with pre, post, synapses, sign."""
    return pd.DataFrame({
        "pre": g.counts.indices, "post": edge_rows(g.counts),
        "synapses": g.counts.data, "sign": g.sign,
    })


def group_edges(g: Graph, labels) -> pd.DataFrame:
    """Collapse neurons into groups (e.g. cell types) and sum the wiring between groups.

    Returns one row per (pre_group, post_group) pair with total synapses, the net signed
    synapses (excitatory minus inhibitory) and the number of neuron-level edges. Neurons whose
    label is missing are dropped.
    """
    labels = pd.Series(labels).reset_index(drop=True)
    e = edge_table(g)
    e["pre_group"] = labels.to_numpy()[e["pre"]]
    e["post_group"] = labels.to_numpy()[e["post"]]
    e = e.dropna(subset=["pre_group", "post_group"])
    e["signed"] = e["synapses"] * e["sign"]
    return (e.groupby(["pre_group", "post_group"], observed=True)
             .agg(synapses=("synapses", "sum"), signed=("signed", "sum"),
                  edges=("synapses", "size"))
             .reset_index()
             .sort_values("synapses", ascending=False, ignore_index=True))


def group_graph(g: Graph, labels) -> tuple[Graph, np.ndarray]:
    """The same wiring at group resolution (e.g. one node per cell type), as a Graph, so every
    function in this module runs unchanged on it. Returns (graph, names) with node k = names[k].

    Why bother: neuron-level strongest paths favour single big connections. A T4 neuron gives
    each LPLC2 only a handful of synapses, so no single T4 -> LPLC2 edge looks strong, even though
    the T4 *population* is a major input. Summing over types restores the population view.

    counts = summed synapses; sign = sign of the net (excitatory - inhibitory) synapses, so a
    group pair whose inputs are mostly modulatory or exactly balanced gets 0.
    """
    ge = group_edges(g, labels)
    names = np.unique(np.concatenate([ge["pre_group"].to_numpy(dtype=str),
                                      ge["post_group"].to_numpy(dtype=str)]))
    pre = np.searchsorted(names, ge["pre_group"].to_numpy(dtype=str))
    post = np.searchsorted(names, ge["post_group"].to_numpy(dtype=str))
    order = np.lexsort((pre, post))                       # rows (post) first, then columns
    pre, post = pre[order], post[order]
    k = len(names)
    indptr = np.zeros(k + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(post, minlength=k))
    counts = sparse.csr_matrix(
        (ge["synapses"].to_numpy(dtype=np.int64)[order], pre, indptr), shape=(k, k)
    )
    sign = np.sign(ge["signed"].to_numpy())[order].astype(np.int8)
    gg = Graph(root_ids=np.arange(k, dtype=np.int64), counts=counts, sign=sign,
               home_neuropil=np.full(k, "NONE"))
    return gg, names


# ========================================================================================
# 5. Probabilistic information-flow layers (Schlegel et al. 2021 eLife; used brain-wide in
#    Schlegel et al. 2024 and Dorkenwald et al. 2024)
# ========================================================================================
def information_flow(g: Graph, seeds, n_runs: int = 256, threshold: float = 0.3,
                     batch_size: int = 64, seed: int = 0,
                     fractions: sparse.csr_matrix | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Mean step at which a random traversal starting at the seeds reaches each neuron.

    Model: at step 0 the seeds are active. A neuron activated at step t gets one chance to
    activate each of its targets at step t+1, succeeding with probability
        p = min(1, w / threshold)       (w = input fraction; threshold 0.3 = 30% of input)
    so an input supplying >=30% of a target's synapses always passes the signal on, and weak
    inputs rarely do. Repeat many times; a neuron's layer is its average activation step.
    Unlike hop distance, a single weak synapse cannot make a neuron look "close".

    Vectorised trick: a target with active inputs j1, j2, ... is NOT activated only if every
    independent trial fails, with probability prod(1 - p_j). In log space that product is a
    sum, which is a sparse matrix product: log P(no) = L @ active, with L[i, j] = log(1 - p_ij).
    We run ``batch_size`` traversals side by side as the columns of a dense (n x batch) matrix.

    Returns (layer, reach): mean activation step over the runs in which the neuron was reached
    (NaN if never), and the fraction of runs that reached it.
    """
    F = input_fractions(g) if fractions is None else fractions
    L = F.copy()
    p = np.minimum(L.data / threshold, 1.0)
    L.data = np.log1p(-np.minimum(p, 1.0 - 1e-12))  # log(1 - p); clip so p = 1 gives ~ -27.6
    rng = np.random.default_rng(seed)

    seed_mask = _as_mask(g.n, seeds)
    step_sum = np.zeros(g.n)
    reached = np.zeros(g.n)
    for start in range(0, n_runs, batch_size):
        R = min(batch_size, n_runs - start)
        visited = np.repeat(seed_mask[:, None], R, axis=1)
        active = visited.astype(np.float64)
        step = np.zeros((g.n, R))
        t = 0
        while active.any():
            t += 1
            p_on = -np.expm1(L @ active)                  # 1 - prod(1 - p) for every (i, run)
            new = (rng.random((g.n, R)) < p_on) & ~visited
            visited |= new
            step[new] = t
            active = new.astype(np.float64)
        step_sum += step.sum(axis=1)                      # seeds contribute 0
        reached += visited.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        layer = np.where(reached > 0, step_sum / reached, np.nan)
    return layer, reached / n_runs
