"""Where on the eye does each neuron look? Columns, an eye map, and receptive fields from wiring.

The compound eye is a hexagonal lattice of ~800 ommatidia per side. Behind every ommatidium sits a
"column": the same set of ~30 columnar cell types (L1-L5, Mi1, Tm2, T4, T5, ...) repeats once per
ommatidium, so a columnar cell's column says exactly which direction in the world it sees.
LPLC2 and LC4 are not columnar: each one collects input from a patch of columns. That patch IS its
receptive field, and the connectome tells us which columns are in it. Three data sets make the
chain from synapses to degrees:

1. Columns (Matsliah et al. 2024, Nature): 22,684 columnar cells of the RIGHT optic lobe of
   FlyWire, each assigned to one of 796 columns with hexagonal coordinates (p, q).
2. Eye map (Zhao et al. 2025, Nature, "Eye structure shapes neuron function"): micro-CT of a whole
   fly head, giving the viewing direction of every ommatidium as a unit vector, on the hexagonal
   grid of the male optic-lobe connectome (Nern et al. 2025). Same grid convention, different fly.
3. Connectivity (the Shiu et al. 2024 table of FlyWire v783, every synapse): who feeds each
   LPLC2 / LC4 neuron.

Both files come from github.com/YijieYin/connectome_interpreter (MIT licence), pinned to a commit
in scripts/download_data.py.

Step A, one eye map for FlyWire (``register_grid``)
--------------------------------------------------
The FlyWire grid (796 columns) and the eye-map grid (852 columns) belong to two different flies,
but both are drawn in the same lattice convention: x = p - q runs front -> back, y = p + q runs
ventral -> dorsal. So we look for the shift (dx, dy) that puts the most FlyWire columns onto
eye-map columns: (2, 38) puts 794 of 796 there. The orientation is checked against anatomy, not
assumed: the lamina is not inverted relative to the retina, and lamina cell bodies (L1-L5) move
posterior as x grows (r = 0.97) and dorsal as y grows (r = -1.00 against FlyWire's ventral-pointing
y). Assumption E1: after this shift a FlyWire column looks where the eye-map column at the same
place looks. Two flies differ by a column or two (~5 deg), small next to receptive fields of
30-60 deg.

Step B, the left optic lobe (``fit_mirror``, ``left_cell_directions``)
---------------------------------------------------------------------
Columns were only assigned on the right. For the left side we use symmetry: reflect every left
columnar cell body into the right optic lobe and give it the direction of its k nearest right
cells of the same type (then mirror that direction, y -> -y, into the left eye). The reflection
is not just "flip x": it is fitted (iterative closest point, a similarity transform with
det = -1) so that left T4s land on right T4s, Mi1s on Mi1s, and so on. Cell bodies are a noisy
handle on a column (leave-one-out on the right: 1-2 columns), but each receptive field averages
hundreds of input cells, so the noise mostly averages out. Assumption E2: the two eyes and optic
lobes are mirror images.

Step C, receptive fields (``receptive_fields``)
-----------------------------------------------
For a neuron n with input synapse counts s_j from mapped columnar cells j looking along unit
vectors d_j:

    center  u_n   = normalise( sum_j s_j d_j )
    size    sigma = sqrt( sum_j s_j angle(d_j, u_n)^2 / sum_j s_j  / 2 )

u_n is where the neuron's input comes from on average; sigma is the spread of that input,
written as the per-axis standard deviation of a 2-D Gaussian (RMS radius / sqrt 2), the form the
encoder uses. Assumption E3: a neuron's receptive field is where its columnar inputs look,
weighted by synapse count.

Coordinates everywhere in the control code: fly frame, x forward, y LEFT, z up. Azimuth
az = atan2(y, x) (0 = straight ahead, +90 = left), elevation el = asin(z).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from c2c.paths import RAW

COLUMNS_FILE = "matsliah2024_fafb_right_columns.csv"
EYEMAP_FILE = "zhao2025_eyemap_right.csv"

# FlyWire annotation positions are voxel coordinates of 4 x 4 x 40 nm. In micrometres:
UM_PER_VOXEL = np.array([0.004, 0.004, 0.040])

# The registration result, measured by register_grid() and checked in the tests.
GRID_SHIFT = (2, 38)


# ---- loading ----------------------------------------------------------------------------------
def load_columns(path: str | Path | None = None) -> pd.DataFrame:
    """Matsliah et al. column assignments: one row per columnar cell (right optic lobe)."""
    path = Path(path) if path is not None else RAW / COLUMNS_FILE
    df = pd.read_csv(path, usecols=["root_id", "type", "column_id", "p", "q"])
    return df.astype({"root_id": "int64", "column_id": "int64", "p": "int64", "q": "int64"})


def load_eyemap(path: str | Path | None = None) -> pd.DataFrame:
    """Zhao et al. eye map: hex coordinates (p, q) -> unit viewing direction (x, y, z), right eye."""
    path = Path(path) if path is not None else RAW / EYEMAP_FILE
    df = pd.read_csv(path, usecols=["x", "y", "z", "p", "q"])
    return df.astype({"p": "int64", "q": "int64"})


def lattice_xy(p, q) -> tuple[np.ndarray, np.ndarray]:
    """Hex coordinates -> lattice coordinates: x = p - q (front -> back), y = p + q (up)."""
    p, q = np.asarray(p), np.asarray(q)
    return p - q, p + q


def az_el(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (..., 3) in the fly frame -> azimuth, elevation in degrees."""
    v = np.asarray(v, dtype=float)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])),
            np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0))))


def unit_vector(az_deg, el_deg) -> np.ndarray:
    """Azimuth, elevation (degrees) -> unit vectors (..., 3)."""
    a, e = np.broadcast_arrays(np.radians(az_deg), np.radians(el_deg))
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, in degrees (broadcasts over leading axes)."""
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))


# ---- step A: register the FlyWire column grid onto the eye map --------------------------------
def register_grid(columns: pd.DataFrame, eyemap: pd.DataFrame, dx_range=range(-10, 11),
                  dy_range=range(80)) -> tuple[int, int, int]:
    """The lattice shift (dx, dy) that lands the most FlyWire columns on eye-map columns.
    Returns (dx, dy, n_columns_overlapping). Only shifts with dx + dy even keep the lattice
    (x + y = 2p is always even), so odd ones are skipped."""
    cols = columns.groupby("column_id")[["p", "q"]].first()
    fx, fy = lattice_xy(cols.p, cols.q)
    ex, ey = lattice_xy(eyemap.p, eyemap.q)
    target = set(zip(ex.tolist(), ey.tolist()))
    best = (0, 0, -1)
    for dx in dx_range:
        for dy in dy_range:
            if (dx + dy) % 2:
                continue
            n = sum((a + dx, b + dy) in target for a, b in zip(fx.tolist(), fy.tolist()))
            if n > best[2]:
                best = (dx, dy, n)
    return best


def column_directions(columns: pd.DataFrame, eyemap: pd.DataFrame,
                      shift=GRID_SHIFT) -> pd.DataFrame:
    """One row per FlyWire column: viewing direction of the RIGHT eye (unit x, y, z), az, el.
    ``on_map`` is False for the few columns that land between eye-map sites; they take the
    direction of the nearest site."""
    cols = columns.groupby("column_id")[["p", "q"]].first()
    fx, fy = lattice_xy(cols.p, cols.q)
    ex, ey = lattice_xy(eyemap.p, eyemap.q)
    tree = cKDTree(np.c_[ex, ey])
    dist, idx = tree.query(np.c_[fx + shift[0], fy + shift[1]])
    d = eyemap[["x", "y", "z"]].to_numpy(float)[idx]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    out = cols.copy()
    out[["x", "y", "z"]] = d
    out["az"], out["el"] = az_el(d)
    out["on_map"] = dist == 0
    return out


# ---- step B: the left optic lobe by mirror symmetry -------------------------------------------
def soma_um(df: pd.DataFrame) -> np.ndarray:
    """Cell-body positions in micrometres (NaN where FlyWire has no soma)."""
    return df[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * UM_PER_VOXEL


MIN_SOMATA = 100   # a type needs this many right cell bodies to serve as a mirror reference


def _right_cells(ann: pd.DataFrame, columns: pd.DataFrame) -> pd.DataFrame:
    """Right columnar cells with a cell body in FlyWire. Photoreceptors (R7, R8) have theirs in
    the retina, outside the brain volume, so types with few somata are dropped."""
    r = columns.merge(ann[["root_id", "soma_x", "soma_y", "soma_z"]], on="root_id", how="left")
    r = r.dropna(subset=["soma_x"])
    n = r["type"].value_counts()
    return r[r["type"].isin(n[n >= MIN_SOMATA].index)].reset_index(drop=True)


def _left_cells(ann: pd.DataFrame, types) -> pd.DataFrame:
    left = ann[ann["side"].eq("left").fillna(False) & ann["cell_type"].isin(types)]
    left = left.dropna(subset=["soma_x"])
    return left[["root_id", "cell_type", "soma_x", "soma_y", "soma_z"]].rename(
        columns={"cell_type": "type"}).reset_index(drop=True)


def fit_mirror(ann: pd.DataFrame, columns: pd.DataFrame, n_iter: int = 30):
    """Reflection that maps the left optic lobe onto the right one.

    Iterative closest point: start from "flip x about the midline", then repeat
      1. match every left cell to the nearest right cell OF THE SAME TYPE
      2. solve for the similarity transform (rotation-with-reflection R, scale s, shift b)
         minimising sum |s R x_left + b - x_right|^2 (orthogonal Procrustes, via an SVD)
    Returns (A, b, residual_um) with A = s R; x_right ~ A @ x_left + b. residual_um is the
    median distance from a mirrored left cell to the nearest right cell of its type."""
    right = _right_cells(ann, columns)
    types = sorted(right["type"].unique())
    left = _left_cells(ann, types)
    xr, xl = soma_um(right), soma_um(left)
    trees = {t: cKDTree(xr[right["type"].to_numpy() == t]) for t in types}
    rows_r = {t: xr[right["type"].to_numpy() == t] for t in types}
    lt = left["type"].to_numpy()

    xmid = 0.5 * (xr[:, 0].mean() + xl[:, 0].mean())
    A, b = np.diag([-1.0, 1.0, 1.0]), np.array([2 * xmid, 0.0, 0.0])
    for _ in range(n_iter):
        y = xl @ A.T + b
        src, dst, dist = [], [], []
        for t in types:
            m = lt == t
            if not m.any():
                continue
            d, i = trees[t].query(y[m])
            src.append(xl[m])
            dst.append(rows_r[t][i])
            dist.append(d)
        s, d = np.vstack(src), np.vstack(dst)
        ms, md = s.mean(0), d.mean(0)
        u, sv, vt = np.linalg.svd((d - md).T @ (s - ms))
        rot = u @ vt                                   # det = -1: a reflection, as it must be
        scale = sv.sum() / ((s - ms) ** 2).sum()
        A, b = scale * rot, md - scale * rot @ ms
    residual = float(np.median(np.concatenate(dist)))
    return A, b, residual


def left_cell_directions(ann: pd.DataFrame, columns: pd.DataFrame, coldirs: pd.DataFrame,
                         A: np.ndarray, b: np.ndarray, k: int = 3) -> pd.DataFrame:
    """Viewing direction of every LEFT columnar cell: mirror its cell body into the right lobe,
    average the right-eye directions of its k nearest same-type neighbours, then reflect the
    direction into the left eye (y -> -y)."""
    right = _right_cells(ann, columns)
    rdir = coldirs.loc[right["column_id"], ["x", "y", "z"]].to_numpy()
    left = _left_cells(ann, sorted(right["type"].unique()))
    y = soma_um(left) @ A.T + b
    out = np.zeros((len(left), 3))
    for t in left["type"].unique():
        mr = right["type"].to_numpy() == t
        ml = left["type"].to_numpy() == t
        _, i = cKDTree(soma_um(right[mr])).query(y[ml], k=k)
        v = rdir[mr][i].mean(axis=1)
        out[ml] = v / np.linalg.norm(v, axis=1, keepdims=True)
    out[:, 1] *= -1.0                                  # right eye -> left eye
    res = left[["root_id", "type"]].copy()
    res[["x", "y", "z"]] = out
    res["side"] = "left"
    return res


def cell_directions(ann: pd.DataFrame, columns: pd.DataFrame | None = None,
                    eyemap: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Viewing direction of every columnar cell on BOTH sides, one row per cell.
    Returns (table with root_id, type, side, x, y, z; diagnostics dict)."""
    columns = load_columns() if columns is None else columns
    eyemap = load_eyemap() if eyemap is None else eyemap
    coldirs = column_directions(columns, eyemap)
    right = columns[["root_id", "type", "column_id"]].copy()
    right[["x", "y", "z"]] = coldirs.loc[right["column_id"], ["x", "y", "z"]].to_numpy()
    right["side"] = "right"
    A, b, residual = fit_mirror(ann, columns)
    left = left_cell_directions(ann, columns, coldirs, A, b)
    table = pd.concat([right.drop(columns="column_id"), left], ignore_index=True)
    diag = {"columns_on_map": int(coldirs["on_map"].sum()), "columns": len(coldirs),
            "mirror_residual_um": residual, "mirror_det": float(np.linalg.det(A)),
            "right_cells": len(right), "left_cells": len(left)}
    return table, diag


# ---- step C: receptive fields from input synapses ---------------------------------------------
def receptive_fields(inputs: pd.DataFrame, cells: pd.DataFrame,
                     targets: pd.DataFrame) -> pd.DataFrame:
    """Receptive field of every target neuron from its columnar inputs.

    inputs   synapse table with columns pre, post, syn (every synapse onto the targets)
    cells    cell_directions() table: root_id, side, x, y, z of mapped columnar cells
    targets  root_id, cell_type, side of the neurons to map (e.g. all LPLC2 and LC4)

    Only inputs from columnar cells on the target's OWN side count (LPLC2 and LC4 dendrites stay
    in the ipsilateral optic lobe). Returns one row per target: x, y, z (unit centre), az, el,
    sigma (deg), syn_mapped, frac_mapped (= mapped / all input synapses, a coverage check)."""
    total = inputs.groupby("post")["syn"].sum()
    m = inputs.merge(cells[["root_id", "side", "x", "y", "z"]], left_on="pre", right_on="root_id")
    m = m.merge(targets[["root_id", "side"]].rename(columns={"root_id": "post",
                                                             "side": "post_side"}), on="post")
    m = m[m["side"] == m["post_side"]]
    rows = []
    for post, g in m.groupby("post"):
        w = g["syn"].to_numpy(float)
        d = g[["x", "y", "z"]].to_numpy()
        c = (w[:, None] * d).sum(0)
        c /= np.linalg.norm(c)
        ang = angle_deg(d, c[None, :])
        sigma = float(np.sqrt((w * ang**2).sum() / w.sum() / 2.0))
        rows.append({"root_id": post, "x": c[0], "y": c[1], "z": c[2], "sigma": sigma,
                     "syn_mapped": int(w.sum()), "frac_mapped": float(w.sum() / total[post])})
    rf = targets.merge(pd.DataFrame(rows), on="root_id", how="left")
    rf["az"], rf["el"] = az_el(rf[["x", "y", "z"]].to_numpy())
    return rf
