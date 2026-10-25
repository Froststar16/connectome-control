"""Tests for c2c.data.annotations: loading, joining onto Graph.root_ids, selection, modalities.

The fake annotation table deliberately has
  * an extra column the loader must ignore (pos_x),
  * empty cells (must become <NA>),
  * a neuron that is annotated but not in the graph (999), and
  * a graph neuron with no annotation row (400),
so the join is tested for order, missing rows and extra rows at once.
"""

import pandas as pd
import pytest

from c2c.data import annotations as an
from c2c.data.graph import build_graph
from c2c.data.loaders import POST, PRE

ROWS = [
    # root_id, flow, super_class, cell_class, cell_sub_class, cell_type, side, top_nt
    (300, "intrinsic", "visual_projection", "", "", "LPLC2", "left", "acetylcholine"),
    (100, "afferent", "sensory", "visual", "", "R1-6", "left", "histamine"),
    (200, "efferent", "descending", "", "", "DNp01", "right", "glutamate"),
    (500, "afferent", "sensory", "mechanosensory", "auditory", "JO-A", "left", "acetylcholine"),
    (600, "afferent", "sensory", "visual", "ocellar", "", "center", "histamine"),
    (700, "afferent", "sensory_ascending", "AN", "SA_DMT_DMetaN", "", "right", "gaba"),
    (999, "intrinsic", "central", "", "", "XYZ", "left", "gaba"),
]


@pytest.fixture
def tsv(tmp_path):
    df = pd.DataFrame(ROWS, columns=["root_id", "flow", "super_class", "cell_class",
                                     "cell_sub_class", "cell_type", "side", "top_nt"])
    df["hemibrain_type"] = ""
    df["nerve"] = ""
    df["pos_x"] = 1.0                       # extra column the loader should not keep
    path = tmp_path / "ann.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def g():
    # A chain over IDs 100..700; 400 is in the graph but has no annotation row.
    ids = [100, 300, 200, 400, 500, 600, 700]
    edges = pd.DataFrame({PRE: ids[:-1], POST: ids[1:], "syn_count": 5, "nt_type": "ACH"})
    return build_graph(edges)


@pytest.fixture
def aligned(tsv, g):
    return an.align_to_graph(an.load_annotations(tsv), g)


def test_load_drops_extra_columns_and_blanks(tsv):
    ann = an.load_annotations(tsv)
    assert list(ann.columns) == an.COLUMNS
    assert ann["root_id"].dtype == "int64"
    assert ann.set_index("root_id").loc[300, "cell_class"] is pd.NA


def test_duplicate_ids_rejected(tsv, tmp_path):
    df = pd.read_csv(tsv, sep="\t")
    dup = tmp_path / "dup.tsv"
    pd.concat([df, df.iloc[:1]]).to_csv(dup, sep="\t", index=False)
    with pytest.raises(ValueError):
        an.load_annotations(dup)


def test_align_is_in_matrix_order(aligned, g):
    assert aligned["root_id"].tolist() == g.root_ids.tolist()   # row k = neuron k
    assert aligned.loc[g.index_of([200])[0], "cell_type"] == "DNp01"
    assert 999 not in aligned["root_id"].tolist()                # not in graph -> dropped


def test_missing_annotation_is_na_not_dropped(aligned, g):
    k = g.index_of([400])[0]
    assert not aligned.loc[k, "annotated"]
    assert an.coverage(aligned) == {
        "graph_neurons": 7, "annotated": 6, "unannotated": 1, "fraction": 6 / 7,
    }


def test_select(aligned, g):
    assert an.select(aligned, cell_type="LPLC2").tolist() == g.index_of([300]).tolist()
    both = an.select(aligned, cell_type=["LPLC2", "DNp01"])
    assert sorted(g.root_ids[both].tolist()) == [200, 300]
    left_sensory = an.select(aligned, super_class="sensory", side="left")
    assert sorted(g.root_ids[left_sensory].tolist()) == [100, 500]
    assert an.select(aligned, cell_type="nope").size == 0


def test_sensory_modality(aligned):
    mod = dict(zip(aligned["root_id"], an.sensory_modality(aligned)))
    assert mod[100] == "visual"
    assert mod[500] == "johnstons_organ"
    assert mod[600] == "ocellar"
    assert mod[700] == "ascending_sensory"
    assert mod[300] is pd.NA and mod[400] is pd.NA               # not sensory / unannotated
