import pandas as pd
import pytest

from c2c.data.loaders import NT_COLS, POST, PRE, collapse_neuropils, threshold


def _row(pre, post, neuropil, syn, nt):
    probs = {c: 0.0 for c in NT_COLS}
    probs[nt] = 1.0
    return {PRE: pre, POST: post, "neuropil": neuropil, "syn_count": syn, **probs}


@pytest.fixture
def raw():
    return pd.DataFrame([
        _row(1, 2, "AL_L", 3, "ach_avg"),   # 1->2 split across two neuropils: 3 + 4 = 7
        _row(1, 2, "LH_L", 4, "ach_avg"),
        _row(2, 3, "MB_L", 10, "gaba_avg"),
        _row(3, 1, "AL_L", 2, "glut_avg"),  # weak edge, dropped at threshold 5
    ])


def test_collapse_sums_over_neuropils(raw):
    edges = collapse_neuropils(raw).set_index([PRE, POST])
    assert edges.loc[(1, 2), "syn_count"] == 7
    assert len(edges) == 3


def test_nt_type_is_argmax(raw):
    edges = collapse_neuropils(raw).set_index([PRE, POST])
    assert edges.loc[(1, 2), "nt_type"] == "ACH"
    assert edges.loc[(2, 3), "nt_type"] == "GABA"


def test_threshold_keeps_split_connection(raw):
    # Without collapsing, neither 1->2 row reaches 5; after collapsing it survives.
    thr = threshold(collapse_neuropils(raw), min_syn=5)
    assert set(zip(thr[PRE], thr[POST])) == {(1, 2), (2, 3)}
