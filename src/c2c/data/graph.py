"""The connectome as a sparse signed matrix.

Convention (used everywhere in this project):

    W[i, j] = signed synaptic input FROM neuron j (presynaptic) TO neuron i (postsynaptic)

Rows are targets, columns are sources. This is the convention of a rate / spiking model,
where the input current to every neuron at once is a matrix-vector product:

    I = W @ r        # r[j] = activity of neuron j, I[i] = total input to neuron i

Sign comes from the predicted neurotransmitter of the presynaptic neuron:
ACH (acetylcholine) is excitatory in the fly brain (+1), GABA and GLUT (glutamate,
which acts mostly through inhibitory chloride channels in the fly CNS) are inhibitory (-1),
and the modulatory transmitters (SER, DA, OCT) get 0 because they do not act as fast
excitation/inhibition.

Two matrices live inside a Graph:
  * ``counts``  unsigned synapse counts for ALL edges (used for degree / weight statistics)
  * ``W``       signed weights; edges with sign 0 are dropped (used for simulation later)
Both share one sparsity pattern, so we store ``counts`` plus a per-edge ``sign`` array and
derive ``W`` on demand. That keeps the .npz file small and the two views consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from c2c.data.loaders import POST, PRE

# Sign of each neurotransmitter class. Anything not listed (SER, DA, OCT) maps to 0.
NT_SIGN: dict[str, int] = {"ACH": +1, "GABA": -1, "GLUT": -1}

# Label for a neuron with no synapses at all in the raw table (should be rare / none).
NO_NEUROPIL = "NONE"


@dataclass
class Graph:
    """A connectome in compressed-sparse-row (CSR) form plus per-neuron metadata.

    Attributes
    ----------
    root_ids : (n,) int64, sorted ascending. ``root_ids[k]`` is the FlyWire root ID of index k.
    counts : (n, n) CSR int32. ``counts[i, j]`` = number of synapses from j onto i.
    sign : int8 array aligned with ``counts.data`` (+1, -1 or 0 for each stored edge).
    home_neuropil : (n,) str. Neuropil where each neuron receives most of its synapses.
    """

    root_ids: np.ndarray
    counts: sparse.csr_matrix
    sign: np.ndarray
    home_neuropil: np.ndarray

    @property
    def n(self) -> int:
        return len(self.root_ids)

    @property
    def W(self) -> sparse.csr_matrix:
        """Signed weight matrix. Same structure as ``counts`` with each value times its sign."""
        W = self.counts.copy()
        W.data = W.data * self.sign  # int32 * int8 -> int32, element-wise per stored edge
        W.eliminate_zeros()          # drop the sign-0 (modulatory) edges from storage
        return W

    def index_of(self, ids) -> np.ndarray:
        """Map FlyWire root IDs -> matrix indices. Raises KeyError for unknown IDs."""
        return ids_to_index(self.root_ids, ids)


# ----------------------------------------------------------------------------------------
# Index mapping
# ----------------------------------------------------------------------------------------
def build_index(edges: pd.DataFrame) -> np.ndarray:
    """Every neuron that appears as pre or post in ``edges``, as a sorted unique array.

    Sorted order matters: it lets us map IDs -> indices with binary search (searchsorted),
    which is vectorised and O(log n) per lookup, instead of a Python dict loop.
    """
    ids = np.concatenate([edges[PRE].to_numpy(), edges[POST].to_numpy()])
    return np.unique(ids)  # np.unique returns the values sorted


def ids_to_index(root_ids: np.ndarray, ids) -> np.ndarray:
    """Vectorised lookup: position of each id in the sorted ``root_ids`` array."""
    ids = np.asarray(ids, dtype=np.int64)
    idx = np.searchsorted(root_ids, ids)  # where each id WOULD go to keep the array sorted
    # searchsorted always returns a position, even for missing IDs, so verify the hit.
    idx_clipped = np.minimum(idx, len(root_ids) - 1)
    found = root_ids[idx_clipped] == ids
    if not np.all(found):
        missing = ids[~found][:5]
        raise KeyError(f"{(~found).sum()} root IDs not in the graph, e.g. {missing.tolist()}")
    return idx


# ----------------------------------------------------------------------------------------
# Matrix construction
# ----------------------------------------------------------------------------------------
def build_counts_and_sign(edges: pd.DataFrame, root_ids: np.ndarray):
    """Build the CSR count matrix and the aligned sign array directly from an edge list.

    ``edges`` needs columns PRE, POST, syn_count, nt_type, with one row per (pre, post) pair
    (that is what loaders.collapse_neuropils + threshold produce).

    We build CSR by hand rather than via COO -> CSR so the order of ``sign`` is guaranteed to
    match the order of ``counts.data``:
      1. sort edges by row (post) then by column (pre)
      2. ``indices`` = column of each edge in that order, ``data`` = its synapse count
      3. ``indptr[i] : indptr[i+1]`` is the slice of edges in row i, computed from row counts
    """
    if edges.duplicated([PRE, POST]).any():
        raise ValueError("edges must have one row per (pre, post) pair; collapse neuropils first")

    n = len(root_ids)
    rows = ids_to_index(root_ids, edges[POST].to_numpy())  # i = postsynaptic = target
    cols = ids_to_index(root_ids, edges[PRE].to_numpy())   # j = presynaptic  = source
    syn = edges["syn_count"].to_numpy(dtype=np.int32)
    sgn = edges["nt_type"].map(NT_SIGN).fillna(0).to_numpy(dtype=np.int8)

    order = np.lexsort((cols, rows))  # lexsort sorts by the LAST key first: rows, then cols
    rows, cols, syn, sgn = rows[order], cols[order], syn[order], sgn[order]

    row_counts = np.bincount(rows, minlength=n)    # number of inputs of each neuron
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_counts)             # running total = start of each row

    counts = sparse.csr_matrix((syn, cols, indptr), shape=(n, n))
    return counts, sgn


def assign_home_neuropil(raw: pd.DataFrame, root_ids: np.ndarray) -> np.ndarray:
    """Home neuropil = the neuropil where a neuron RECEIVES the most synapses.

    Uses the raw per-neuropil table (all proofread connections, before thresholding), because
    weak connections still tell us where a neuron's dendrites are. Neurons with no input at
    all (e.g. some sensory neurons whose inputs are outside the brain) fall back to the
    neuropil where they SEND the most synapses.
    """
    home = pd.Series(NO_NEUROPIL, index=root_ids, dtype=object)

    # Fill outputs first, then overwrite with inputs, so inputs take priority.
    for id_col in (PRE, POST):
        per_np = (
            raw.groupby([id_col, "neuropil"], observed=True)["syn_count"].sum().reset_index()
        )
        # Sort by count (desc) with neuropil name as a tie-breaker so the result is
        # deterministic, then keep the first (largest) row per neuron.
        per_np = per_np.sort_values(["syn_count", "neuropil"], ascending=[False, True])
        best = per_np.drop_duplicates(id_col).set_index(id_col)["neuropil"].astype(str)
        best = best[best.index.isin(root_ids)]
        home.loc[best.index] = best.to_numpy()

    return home.to_numpy(dtype=str)


def build_graph(edges: pd.DataFrame, raw: pd.DataFrame | None = None) -> Graph:
    """Full pipeline: thresholded edge list (+ raw table for home neuropils) -> Graph."""
    root_ids = build_index(edges)
    counts, sign = build_counts_and_sign(edges, root_ids)
    if raw is None:
        home = np.full(len(root_ids), NO_NEUROPIL)
    else:
        home = assign_home_neuropil(raw, root_ids)
    return Graph(root_ids=root_ids, counts=counts, sign=sign, home_neuropil=home)


# ----------------------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------------------
def save_graph(g: Graph, path: str | Path) -> Path:
    """Store the CSR arrays + metadata in one compressed .npz (no pickle needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        root_ids=g.root_ids,
        indptr=g.counts.indptr,
        indices=g.counts.indices,
        data=g.counts.data,
        sign=g.sign,
        shape=np.array(g.counts.shape),
        home_neuropil=g.home_neuropil.astype(str),
    )
    return path


def load_graph(path: str | Path) -> Graph:
    """Inverse of save_graph."""
    with np.load(path, allow_pickle=False) as f:  # allow_pickle=False: never execute code
        counts = sparse.csr_matrix(
            (f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"])
        )
        return Graph(
            root_ids=f["root_ids"],
            counts=counts,
            sign=f["sign"],
            home_neuropil=f["home_neuropil"],
        )
