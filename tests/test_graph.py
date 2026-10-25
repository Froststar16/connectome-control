"""Tests for c2c.data.graph and c2c.analysis.graph_stats on a hand-made 4-neuron connectome.

Root IDs are deliberately large and unsorted (like real FlyWire IDs) so the tests catch
mistakes in the ID -> index mapping.

    neuron 900 (ACH)  --10-->  neuron 300      excitatory
    neuron 900 (ACH)  --6--->  neuron 700      excitatory
    neuron 300 (GABA) --8--->  neuron 700      inhibitory
    neuron 700 (GLUT) --5--->  neuron 100      inhibitory
    neuron 100 (SER)  --7--->  neuron 900      modulatory -> sign 0, absent from W

Sorted IDs [100, 300, 700, 900] -> indices [0, 1, 2, 3].
"""

import numpy as np
import pandas as pd
import pytest

from c2c.analysis import graph_stats as gs
from c2c.data.graph import build_graph, ids_to_index, load_graph, save_graph
from c2c.data.loaders import POST, PRE


@pytest.fixture
def edges():
    return pd.DataFrame({
        PRE:  [900, 900, 300, 700, 100],
        POST: [300, 700, 700, 100, 900],
        "syn_count": [10, 6, 8, 5, 7],
        "nt_type": ["ACH", "ACH", "GABA", "GLUT", "SER"],
    })


@pytest.fixture
def raw():
    # Per-neuropil rows. Neuron 700 receives 6 in AL_L, 8 split as 2 AL_L + 6 MB_L
    # -> AL_L total 8 vs MB_L 6 -> home AL_L. Neuron 300 receives only in MB_R.
    return pd.DataFrame({
        PRE:  [900, 900, 300, 300, 700, 100],
        POST: [300, 700, 700, 700, 100, 900],
        "neuropil": ["MB_R", "AL_L", "AL_L", "MB_L", "LH_R", "GNG"],
        "syn_count": [10, 6, 2, 6, 5, 7],
    })


@pytest.fixture
def g(edges, raw):
    return build_graph(edges, raw)


# ---- index mapping ---------------------------------------------------------------------
def test_index_is_sorted_unique(g):
    assert g.root_ids.tolist() == [100, 300, 700, 900]


def test_ids_to_index_roundtrip(g):
    ids = np.array([900, 100, 700])
    idx = g.index_of(ids)
    assert idx.tolist() == [3, 0, 2]
    assert g.root_ids[idx].tolist() == ids.tolist()


def test_unknown_id_raises(g):
    with pytest.raises(KeyError):
        ids_to_index(g.root_ids, [100, 555])


# ---- matrix shape, orientation and signs -----------------------------------------------
def test_shape_and_counts(g):
    assert g.counts.shape == (4, 4)
    assert g.counts.nnz == 5                 # every edge, including the modulatory one
    assert g.counts.sum() == 10 + 6 + 8 + 5 + 7


def test_orientation_row_is_target(g):
    i300, i900 = g.index_of([300, 900])
    # 900 -> 300 with 10 synapses lives at row = target (300), column = source (900)
    assert g.counts[i300, i900] == 10
    assert g.counts[i900, i300] == 0


def test_signs(g):
    W = g.W
    ix = dict(zip([100, 300, 700, 900], range(4)))
    assert W[ix[300], ix[900]] == +10        # ACH  -> excitatory
    assert W[ix[700], ix[300]] == -8         # GABA -> inhibitory
    assert W[ix[100], ix[700]] == -5         # GLUT -> inhibitory
    assert W[ix[900], ix[100]] == 0          # SER  -> dropped
    assert W.nnz == 4


def test_duplicate_pairs_rejected(edges, raw):
    with pytest.raises(ValueError):
        build_graph(pd.concat([edges, edges.iloc[:1]]), raw)


# ---- home neuropil ---------------------------------------------------------------------
def test_home_neuropil(g):
    home = dict(zip(g.root_ids.tolist(), g.home_neuropil.tolist()))
    assert home[700] == "AL_L"   # 8 input synapses in AL_L beats 6 in MB_L
    assert home[300] == "MB_R"
    assert home[100] == "LH_R"
    assert home[900] == "GNG"


# ---- save / load -----------------------------------------------------------------------
def test_save_load_roundtrip(g, tmp_path):
    g2 = load_graph(save_graph(g, tmp_path / "g.npz"))
    assert np.array_equal(g2.root_ids, g.root_ids)
    assert (g2.counts != g.counts).nnz == 0
    assert (g2.W != g.W).nnz == 0
    assert g2.home_neuropil.tolist() == g.home_neuropil.tolist()


# ---- statistics ------------------------------------------------------------------------
def test_degrees(g):
    # order: 100, 300, 700, 900
    assert gs.in_degree(g).tolist() == [1, 1, 2, 1]
    assert gs.out_degree(g).tolist() == [1, 1, 1, 2]


def test_neuropil_matrix_conserves_synapses(g):
    M, names = gs.neuropil_matrix(g, merge=True)
    assert M.sum() == g.counts.sum()
    by_name = {n: i for i, n in enumerate(names)}
    # 900 (GNG) -> 300 (MB) carries 10 synapses
    assert M[by_name["GNG"], by_name["MB"]] == 10


def test_ei_share(g):
    t = gs.ei_share_by_region(g, merge=True)
    assert t.loc["AL", "excitatory"] == 6 and t.loc["AL", "inhibitory"] == 8
    assert t.loc["GNG", "modulatory_share"] == 1.0
    shares = t[["excitatory_share", "inhibitory_share", "modulatory_share"]].sum(axis=1)
    assert np.allclose(shares, 1.0)


def test_merge_hemispheres():
    assert gs.merge_hemispheres(np.array(["AL_L", "MB_CA_R", "GNG"])).tolist() == \
        ["AL", "MB_CA", "GNG"]
