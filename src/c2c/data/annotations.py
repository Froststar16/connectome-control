"""Neuron annotations (what each neuron IS) joined onto the connectome graph (who it talks to).

Source: the FlyWire hierarchical annotations of Schlegel et al. 2024 and Dorkenwald et al. 2024,
maintained at github.com/flyconnectome/flywire_annotations. We pin release v3.2.0, which is
keyed to the same FlyWire materialization (783) as our connectivity table, so root IDs match.
(These labels are NOT in the Zenodo connectivity record; Codex serves similar tables but needs a
login token, so the GitHub release is the reproducible choice.)

The hierarchy we use, from coarse to fine:

    flow         afferent (into the brain) / intrinsic / efferent (out of the brain)
    super_class  sensory, optic, central, visual_projection, descending, ascending, motor, ...
    cell_class   e.g. visual, mechanosensory, olfactory (for sensory neurons)
    cell_sub_class  e.g. auditory, wind_gravity, ocellar, sugar
    cell_type    e.g. LPLC2, LC4, DNp01 (the giant fiber)
    side         left / right / center

Joining strategy: ``Graph.root_ids`` is the sorted list of neurons in our thresholded graph, and
matrix index k corresponds to ``root_ids[k]``. ``pandas.reindex(root_ids)`` returns a table with
exactly one row per graph neuron, in matrix order, so row k of the result describes neuron k.
Neurons with no annotation row become NaN instead of silently disappearing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2c.data.graph import Graph
from c2c.paths import RAW

ANNOTATION_VERSION = "v3.2.0"
ANNOTATION_FILE = f"neuron_annotations_{ANNOTATION_VERSION}.tsv"

# Columns we keep. The TSV has ~30; positions, lineages and cross-references are not needed here.
COLUMNS = [
    "root_id", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
    "hemibrain_type", "side", "nerve", "top_nt",
]

# super_class values that count as sensory input to the brain.
SENSORY_SUPER_CLASSES = ("sensory", "sensory_ascending")


def load_annotations(path: str | Path | None = None) -> pd.DataFrame:
    """Read the annotation TSV. Empty cells become <NA>, so ``.isna()`` means "no label"."""
    path = Path(path) if path is not None else RAW / ANNOTATION_FILE
    ann = pd.read_csv(
        path, sep="\t", usecols=COLUMNS,
        dtype={c: "string" for c in COLUMNS if c != "root_id"} | {"root_id": "int64"},
    )[COLUMNS]  # usecols keeps the FILE's column order; index with COLUMNS to fix ours
    # Some cells are empty strings rather than missing; treat both the same way.
    text_cols = [c for c in COLUMNS if c != "root_id"]
    ann[text_cols] = ann[text_cols].replace("", pd.NA)
    if not ann["root_id"].is_unique:
        raise ValueError("annotation table has duplicate root_ids")
    return ann


def align_to_graph(ann: pd.DataFrame, g: Graph) -> pd.DataFrame:
    """One row per graph neuron, in matrix-index order (row k describes neuron k).

    The result keeps ``root_id`` as a column and uses a plain 0..n-1 index, so
    ``aligned.loc[k]`` and ``g.counts[k]`` refer to the same neuron.
    """
    aligned = ann.set_index("root_id").reindex(g.root_ids)
    aligned.index.name = "root_id"
    aligned = aligned.reset_index()
    aligned["annotated"] = aligned["super_class"].notna()
    return aligned


def coverage(aligned: pd.DataFrame) -> dict:
    """How many graph neurons got a label. Worth printing every time: a join that silently
    matches 0 rows (wrong ID dtype, wrong materialization) looks exactly like 'no annotation'."""
    n = len(aligned)
    hit = int(aligned["annotated"].sum())
    return {"graph_neurons": n, "annotated": hit, "unannotated": n - hit, "fraction": hit / n}


def select(aligned: pd.DataFrame, **criteria) -> np.ndarray:
    """Matrix indices of neurons matching every criterion.

    A criterion value can be a single label or a list of labels, e.g.
        select(aligned, cell_type="LPLC2")
        select(aligned, super_class="descending", side="left")
        select(aligned, cell_type=["LPLC2", "LC4"])
    """
    mask = np.ones(len(aligned), dtype=bool)
    for column, wanted in criteria.items():
        values = [wanted] if isinstance(wanted, str) else list(wanted)
        mask &= aligned[column].isin(values).to_numpy(dtype=bool, na_value=False)
    return np.flatnonzero(mask)


def sensory_modality(aligned: pd.DataFrame) -> pd.Series:
    """A modality label for every sensory neuron (<NA> for everything else).

    Mostly ``cell_class``, with three splits that matter for behaviour:
      * ocellar photoreceptors (the three simple eyes on top of the head) vs compound-eye ones
      * Johnston's organ (auditory + wind/gravity, in the antenna) vs other mechanosensors
      * 'sensory_ascending' neurons, whose cell bodies sit in the ventral nerve cord, as their own
        group, because their modality is mixed (mostly leg/body sensors relayed up the neck)
    """
    sc = aligned["super_class"]
    cls = aligned["cell_class"].fillna("unknown_sensory")
    sub = aligned["cell_sub_class"].fillna("")

    modality = pd.Series(pd.NA, index=aligned.index, dtype="string")
    is_sensory = sc.eq("sensory").fillna(False)
    modality[is_sensory] = cls[is_sensory].replace({"optic_lobes": "unknown_sensory"})
    modality[is_sensory & (sub == "ocellar")] = "ocellar"
    modality[is_sensory & sub.isin(["auditory", "wind_gravity"])] = "johnstons_organ"
    modality[sc.eq("sensory_ascending").fillna(False)] = "ascending_sensory"
    return modality
