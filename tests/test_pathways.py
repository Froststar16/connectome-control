"""Tests for c2c.analysis.pathways on a hand-made 7-neuron circuit with known answers.

    S --10 ACH--> A --20 GABA--> D --6 SER--> F
    S --5  ACH--> B --8 ACH--> C --30 ACH--> D
                  E --50 ACH--> D
                  E --10 GLUT-> A

IDs are 100 (S), 200 (A), 300 (B), 400 (D), 500 (C), 600 (E), 700 (F), so the matrix indices are
S=0, A=1, B=2, D=3, C=4, E=5, F=6 (sorted by ID). Total input per neuron: A 20, B 5, C 8,
D 100, F 6, so the input fractions are:

    S->A 0.5   E->A 0.5   S->B 1.0   B->C 1.0   A->D 0.2   C->D 0.3   E->D 0.5   D->F 1.0

What the circuit is built to test:
  * shortest path S->A->D (2 hops) is NOT the strongest: S->B->C->D has strength 1*1*0.3 = 0.3
    versus 0.5*0.2 = 0.1, so Dijkstra on -log(fraction) must take the long way round;
  * S->B, B->C and D->F have fraction exactly 1, i.e. cost -log(1) = 0 (the MIN_COST edge case);
  * E->A->D is GLUT then GABA: two inhibitions = disinhibition, net sign +1;
  * D->F is serotonergic (sign 0), so any chain through it has zero signed effect;
  * E is upstream of D but unreachable from S.
"""

import numpy as np
import pandas as pd
import pytest

from c2c.analysis import pathways as pw
from c2c.data.graph import build_graph
from c2c.data.loaders import POST, PRE

S, A, B, D, C, E, F = range(7)


@pytest.fixture
def g():
    edges = pd.DataFrame({
        PRE:  [100, 100, 200, 300, 500, 600, 600, 400],
        POST: [200, 300, 400, 500, 400, 400, 200, 700],
        "syn_count": [10, 5, 20, 8, 30, 50, 10, 6],
        "nt_type": ["ACH", "ACH", "GABA", "ACH", "ACH", "ACH", "GLUT", "SER"],
    })
    return build_graph(edges)


def test_index_layout(g):
    assert g.root_ids.tolist() == [100, 200, 300, 400, 500, 600, 700]


# ---- input fractions -------------------------------------------------------------------
def test_fractions_rows_sum_to_one_and_keep_order(g):
    F_ = pw.input_fractions(g)
    sums = np.asarray(F_.sum(axis=1)).ravel()
    has_input = np.diff(g.counts.indptr) > 0
    assert np.allclose(sums[has_input], 1.0)
    # same storage order as counts, so F.data[e] belongs to the same edge as g.sign[e]
    assert np.array_equal(F_.indices, g.counts.indices)
    assert np.array_equal(F_.indptr, g.counts.indptr)
    assert F_[D, C] == pytest.approx(0.3)
    assert F_[A, S] == pytest.approx(0.5)


# ---- BFS -------------------------------------------------------------------------------
def test_hop_distance_downstream(g):
    d = pw.hop_distance(g, [S])
    assert d.tolist() == [0, 1, 1, 2, 2, pw.UNREACHED, 3]   # S A B D C E F


def test_hop_distance_upstream(g):
    d = pw.hop_distance(g, [D], direction="upstream")
    assert d[D] == 0
    assert d[[A, C, E]].tolist() == [1, 1, 1]
    assert d[[S, B]].tolist() == [2, 2]
    assert d[F] == pw.UNREACHED                              # F is downstream only


def test_hop_distance_multi_source_and_max_hops(g):
    d = pw.hop_distance(g, [S, E], max_hops=1)
    assert d[[S, E]].tolist() == [0, 0]
    assert d[[A, B, D]].tolist() == [1, 1, 1]                # D is 1 hop from E
    assert d[C] == pw.UNREACHED and d[F] == pw.UNREACHED     # beyond max_hops


def test_hop_layers(g):
    layers = pw.hop_layers(pw.hop_distance(g, [S]))
    assert {h: v.tolist() for h, v in layers.items()} == {0: [S], 1: [A, B], 2: [D, C], 3: [F]}


def test_bad_arguments(g):
    with pytest.raises(ValueError):
        pw.hop_distance(g, [])
    with pytest.raises(ValueError):
        pw.hop_distance(g, [S], direction="sideways")


# ---- strongest paths -------------------------------------------------------------------
def test_strongest_path_prefers_strong_long_route(g):
    sp = pw.strongest_paths(g, [S])
    assert sp.path_to(D) == [S, B, C, D]                     # not the 2-hop S->A->D
    assert sp.strength[D] == pytest.approx(0.3, rel=1e-9)
    assert sp.path_to(F) == [S, B, C, D, F]


def test_strongest_paths_zero_cost_edges_and_unreachable(g):
    sp = pw.strongest_paths(g, [S])
    assert sp.strength[C] == pytest.approx(1.0)              # S->B->C, both fraction 1
    assert sp.path_to(C) == [S, B, C]
    assert not np.isfinite(sp.cost[E])
    assert sp.path_to(E) == []


def test_strongest_paths_multi_source(g):
    sp = pw.strongest_paths(g, [S, E])
    assert sp.path_to(D) == [E, D]                           # 0.5 beats 0.3
    assert sp.source[D] == E


# ---- signed effects --------------------------------------------------------------------
def test_path_effect_excitatory_chain(g):
    pe = pw.path_effect(g, [S, B, C, D])
    assert pe.synapses.tolist() == [5, 8, 30]
    assert pe.sign.tolist() == [1, 1, 1]
    assert pe.effect == pytest.approx(0.3)


def test_path_effect_inhibitory_and_disinhibition(g):
    assert pw.path_effect(g, [S, A, D]).effect == pytest.approx(-0.1)   # + then -
    dis = pw.path_effect(g, [E, A, D])                                  # - then -
    assert dis.net_sign == +1
    assert dis.effect == pytest.approx(0.1)


def test_path_effect_modulatory_zeroes_chain(g):
    pe = pw.path_effect(g, [C, D, F])
    assert pe.net_sign == 0 and pe.effect == 0


def test_missing_edge_raises(g):
    with pytest.raises(KeyError):
        pw.path_effect(g, [S, D])


def test_multihop_effect(g):
    x = pw.multihop_effect(g, [S], n_hops=3)
    assert x.shape == (4, 7)
    assert x[1, A] == pytest.approx(0.5) and x[1, B] == pytest.approx(1.0)
    assert x[2, D] == pytest.approx(-0.1)                    # via A (inhibitory)
    assert x[3, D] == pytest.approx(0.3)                     # via B, C (excitatory)
    assert x[3, F] == 0                                      # serotonergic last step
    unsigned = pw.multihop_effect(g, [S], n_hops=3, signed=False)
    assert unsigned[3, F] == pytest.approx(0.1)              # S->A->D->F = 0.5*0.2*1


# ---- subgraph and grouping -------------------------------------------------------------
def test_subgraph_keeps_counts_and_signs_aligned(g):
    sub = pw.subgraph(g, [E, D, A, S])                       # order should not matter
    assert sub.root_ids.tolist() == [100, 200, 400, 600]     # S A D E
    assert sub.counts.nnz == 4
    s, a, d, e = range(4)
    W = sub.W
    assert W[a, s] == 10 and W[d, a] == -20 and W[d, e] == 50 and W[a, e] == -10


def test_group_edges(g):
    labels = ["sens", "inter", "inter", "DN", "inter", "other", None]   # by index
    ge = pw.group_edges(g, labels).set_index(["pre_group", "post_group"])
    row = ge.loc[("inter", "DN")]
    assert (row.synapses, row.signed, row.edges) == (50, 10, 2)          # A->D -20, C->D +30
    assert ge.loc[("inter", "inter")].synapses == 8
    # D's only output goes to the unlabelled F, so no row starts at "DN"
    assert "DN" not in ge.index.get_level_values("pre_group")


def test_group_graph_runs_the_same_algorithms(g):
    labels = ["sens", "inter", "inter", "DN", "inter", "other", None]
    gg, names = pw.group_graph(g, labels)
    assert names.tolist() == ["DN", "inter", "other", "sens"]
    ix = {name: k for k, name in enumerate(names)}
    assert gg.counts[ix["inter"], ix["sens"]] == 15          # S->A 10 + S->B 5
    assert gg.counts[ix["inter"], ix["inter"]] == 8          # B->C, a within-group edge
    W = gg.W
    assert W[ix["DN"], ix["inter"]] == 50                    # net +10 -> sign +1
    assert W[ix["inter"], ix["other"]] == -10                # E->A is glutamatergic
    sp = pw.strongest_paths(gg, [ix["sens"]])
    assert sp.path_to(ix["DN"]) == [ix["sens"], ix["inter"], ix["DN"]]
    assert sp.strength[ix["DN"]] == pytest.approx(15 / 33 * 50 / 100)


# ---- information flow ------------------------------------------------------------------
def test_information_flow_matches_hand_calculation(g):
    # p = min(1, w / 0.3): S->A, S->B, B->C, C->D, D->F all have p = 1; A->D has p = 0.2/0.3.
    # So A, B always step 1, C step 2, and D is step 2 with prob 2/3 (via A) else step 3:
    # expected layer(D) = 2*(2/3) + 3*(1/3) = 7/3, and F is always one step after D.
    layer, reach = pw.information_flow(g, [S], n_runs=4000, batch_size=500, seed=1)
    assert layer[S] == 0 and reach[S] == 1
    assert layer[A] == pytest.approx(1.0) and layer[B] == pytest.approx(1.0)
    assert layer[C] == pytest.approx(2.0)
    assert layer[D] == pytest.approx(7 / 3, abs=0.04)
    assert layer[F] == pytest.approx(10 / 3, abs=0.04)
    assert np.isnan(layer[E]) and reach[E] == 0


def test_information_flow_is_reproducible(g):
    a = pw.information_flow(g, [S], n_runs=64, seed=7)[0]
    b = pw.information_flow(g, [S], n_runs=64, seed=7)[0]
    assert np.array_equal(a, b, equal_nan=True)
