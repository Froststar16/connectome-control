"""The connectome exactly as Shiu et al. (Nature 2024) fed it to their model, as a ``Graph``.

Their repository (github.com/philshiu/Drosophila_brain_model) ships ``Connectivity_783.parquet``:
the same FlyWire v783 synapse table we use, collapsed over neuropils, but with two differences
from our Phase 1 graph:

* NO threshold: every pair with at least 1 synapse is kept (15.1 million pairs vs our 2.7 million
  pairs with >= 5 synapses). Pairs with 1-4 synapses are 82% of pairs but only ~37% of synapses.
* ONE SIGN PER NEURON (Dale's law): a neuron is inhibitory if most of its presynaptic sites are
  predicted GABA or glutamate, otherwise excitatory. Dopamine, serotonin and octopamine neurons
  therefore count as EXCITATORY. We instead sign each edge from that edge's own transmitter
  prediction and give modulatory edges 0.

Loading it as a Graph lets every simulation and analysis function run on both versions, so the
effect of each difference can be measured instead of argued about.

The sugar-neuron list and the MN9 ID are the ones in the authors' notebooks (example.ipynb,
figures.ipynb). They were picked on FlyWire materialization 630; 20 of the 21 sugar IDs and the
MN9 ID survive unchanged in v783 (one sugar neuron was edited and got a new ID).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2c.data.graph import Graph, build_graph
from c2c.data.loaders import POST, PRE
from c2c.paths import RAW

SHIU_CONNECTIVITY = "shiu2024_connectivity_783.parquet"

# Labellar sugar-sensing gustatory receptor neurons (GRNs), one side, from the authors' notebooks.
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
# MN9, the motor neuron of proboscis extension that the paper uses as its readout.
MN9 = 720575940660219265

# MN9 firing in the authors' own published output (results/example/*.parquet in their repo,
# materialization 630, 30 trials x 1 s): our reference numbers for the validation.
SHIU_REFERENCE_MN9_HZ = {100: 67.0, 200: 93.3}
SHIU_REFERENCE_ACTIVE = {100: 404, 200: 448}


def present(g: Graph, ids) -> np.ndarray:
    """The subset of ``ids`` that exist in ``g`` (in the given order)."""
    ids = np.asarray(ids, dtype=np.int64)
    return ids[np.isin(ids, g.root_ids)]


def load_shiu_graph(path: str | Path | None = None, min_syn: int = 1) -> Graph:
    """Shiu et al.'s connectivity table -> Graph with their per-neuron signs.

    The sign is passed through ``build_graph`` by labelling excitatory edges 'ACH' and inhibitory
    ones 'GABA', which map to +1 and -1. (Those labels are stand-ins for the sign, not their
    transmitter predictions.) ``min_syn=5`` gives their signs on our thresholded edge set.
    """
    path = Path(path) if path is not None else RAW / SHIU_CONNECTIVITY
    con = pd.read_parquet(path, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity",
                                         "Excitatory"])
    con = con[con["Connectivity"] >= min_syn]
    edges = pd.DataFrame({
        PRE: con["Presynaptic_ID"].to_numpy(np.int64),
        POST: con["Postsynaptic_ID"].to_numpy(np.int64),
        "syn_count": con["Connectivity"].to_numpy(np.int32),
        "nt_type": np.where(con["Excitatory"].to_numpy() > 0, "ACH", "GABA"),
    })
    return build_graph(edges)
