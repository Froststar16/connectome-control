"""The Phase 5 experiments: where the threat is, and which way the fly goes.

Phase 4 drove every LPLC2 (or LC4) neuron on one side with the same rate and sent the same command
to both sides of the body, so it could only test a frontal threat and a straight back-up. Here:

* each LPLC2 / LC4 neuron gets its own receptive field, measured from the connectome
  (c2c.data.eyemap) and read off flygym's compound eye (RetinotopicEncoder);
* the decoder steers: left/right asymmetry of the escape DNs turns the fly (SteeringDecoder);
* the ball comes from several azimuths, and every run is done with both encoders.

Everything slow is cached under data/processed/phase5/ (delete a file or pass force=True to
re-run it); ``scripts/run_phase5.py`` runs it all from a terminal.

Runs
----
<encoder>_az<azimuth>_s<seed>   closed loop, steering decoder, encoder "uniform" (Phase 4) or
                                "retinotopic", ball from that azimuth (deg, + = left)
baseline_walk                   brain disconnected, no ball: where the fly walks if it never
                                reacts (used to aim the ball, see ``aimed_ball``)
record_<encoder>_az<azimuth>    standing fly, no brain: what each of the 314 input neurons
                                receives (per-neuron drive logged); replayed into the brain
                                alone by ``replay_table``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from c2c.analysis import phase4 as p4
from c2c.analysis.phase3 import network
from c2c.control.body import FlyArena, LoomingBall
from c2c.control.decoder import EscapeDecoder, SteeringDecoder
from c2c.control.encoder import LoomingEncoder, looming_drive, retinotopic_from_table
from c2c.control.loop import ClosedLoop
from c2c.paths import PROCESSED

CACHE = PROCESSED / "phase5"
VARIANT = "shiu_min5"                 # Phase 4's choice: per-neuron signs, >= 5 synapses
AZIMUTHS = (0, 45, -45, 90, -90)
ENCODERS = ("uniform", "retinotopic")
SEEDS = (0, 1, 2)
T_RUN = 1000.0
BALL = dict(p4.BALL)                  # 60 mm away, 20 deg up, 150 mm/s from t = 100 ms
ESCAPE_TYPES = ("DNp01", "DNp02", "DNp04", "DNp11")
GIF_AZIMUTH = 90
ESCAPE_WINDOW = 300.0                 # ms after input onset over which the escape direction is read
LPLC2_RF_FWHM = 60.0                  # deg, measured LPLC2 receptive field (Klapoetke et al. 2017)
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))      # 2.355: Gaussian full width at half max


# ---- receptive fields and eye geometry (cached) ------------------------------------------------
def receptive_field_table(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """RF of every LPLC2 and LC4 neuron (both sides) + diagnostics. Cached as CSV + JSON."""
    from c2c.data import eyemap, shiu
    from c2c.data.annotations import ANNOTATION_FILE, RAW

    path, dpath = CACHE / "receptive_fields.csv", CACHE / "receptive_fields.json"
    if path.exists() and dpath.exists() and not force:
        return pd.read_csv(path), json.loads(dpath.read_text())
    ann = pd.read_csv(RAW / ANNOTATION_FILE, sep="\t", low_memory=False,
                      usecols=["root_id", "cell_type", "side", "soma_x", "soma_y", "soma_z"])
    cells, diag = eyemap.cell_directions(ann)
    targets = ann[ann["cell_type"].isin(["LPLC2", "LC4"])][["root_id", "cell_type", "side"]]
    inputs = shiu.input_synapses(targets["root_id"])
    rf = eyemap.receptive_fields(inputs, cells, targets)
    CACHE.mkdir(parents=True, exist_ok=True)
    rf.to_csv(path, index=False)
    dpath.write_text(json.dumps(diag, indent=1))
    return rf, diag


def mirror_validation(k: int = 3) -> dict:
    """How good is the left side's cell-body shortcut? Run it on the RIGHT side, where the true
    columns are known: give each right columnar cell the direction of its k nearest OTHER
    same-type cells by cell body, rebuild the right LPLC2/LC4 receptive fields from those
    estimates, and compare with the ones from true columns."""
    from scipy.spatial import cKDTree

    from c2c.data import eyemap, shiu
    from c2c.data.annotations import ANNOTATION_FILE, RAW

    ann = pd.read_csv(RAW / ANNOTATION_FILE, sep="\t", low_memory=False,
                      usecols=["root_id", "cell_type", "side", "soma_x", "soma_y", "soma_z"])
    cols = eyemap.load_columns()
    coldirs = eyemap.column_directions(cols, eyemap.load_eyemap())
    right = eyemap._right_cells(ann, cols)
    true_dir = coldirs.loc[right["column_id"], ["x", "y", "z"]].to_numpy()
    est = np.zeros_like(true_dir)
    for t in right["type"].unique():
        m = right["type"].to_numpy() == t
        pos = eyemap.soma_um(right[m])
        _, i = cKDTree(pos).query(pos, k=k + 1)
        v = true_dir[m][i[:, 1:]].mean(axis=1)                   # leave the cell itself out
        est[m] = v / np.linalg.norm(v, axis=1, keepdims=True)
    targets = ann[ann["cell_type"].isin(["LPLC2", "LC4"]) & ann["side"].eq("right")][
        ["root_id", "cell_type", "side"]]
    inputs = shiu.input_synapses(targets["root_id"])
    truth = right[["root_id", "type"]].assign(side="right")
    truth[["x", "y", "z"]] = true_dir
    guess = truth.copy()
    guess[["x", "y", "z"]] = est
    rf_t = eyemap.receptive_fields(inputs, truth, targets).set_index("root_id")
    rf_g = eyemap.receptive_fields(inputs, guess, targets).set_index("root_id").loc[rf_t.index]
    err = eyemap.angle_deg(rf_t[["x", "y", "z"]].to_numpy(), rf_g[["x", "y", "z"]].to_numpy())
    return {"cell_error_deg_median": float(np.median(eyemap.angle_deg(est, true_dir))),
            "rf_centre_error_deg_median": float(np.median(err)),
            "rf_centre_error_deg_max": float(err.max()),
            "sigma_true_mean": float(rf_t["sigma"].mean()),
            "sigma_estimated_mean": float(rf_g["sigma"].mean())}


def ommatidia_dirs(force: bool = False) -> np.ndarray:
    """flygym's ommatidium viewing directions (2, 721, 3), head frame. Needs flygym once."""
    path = CACHE / "ommatidia_dirs.npy"
    if path.exists() and not force:
        return np.load(path)
    arena = FlyArena()
    try:
        d = arena.ommatidia_directions()
    finally:
        arena.close()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, d)
    return d


def sigma_scale(rf: pd.DataFrame, fwhm_deg: float = LPLC2_RF_FWHM) -> float:
    """Assumption R3: scale connectome RF widths so the median LPLC2 RF has the measured width."""
    return float(fwhm_deg / FWHM_PER_SIGMA / rf.loc[rf["cell_type"] == "LPLC2", "sigma"].median())


def make_encoder(kind: str, root_ids=None):
    """"uniform" -> Phase 4's LoomingEncoder; "retinotopic" -> RetinotopicEncoder for the
    neurons ``root_ids`` (in drive order)."""
    if kind == "uniform":
        return LoomingEncoder()
    if kind == "retinotopic":
        rf, _ = receptive_field_table()
        return retinotopic_from_table(rf, root_ids, ommatidia_dirs(),
                                      sigma_scale=sigma_scale(rf))
    raise ValueError(f"unknown encoder {kind!r}; choose from {ENCODERS}")


# ---- one run ------------------------------------------------------------------------------------
def baseline_walk(force: bool = False) -> dict:
    """The fly walking with no brain and no ball: where it would be at every moment if it never
    reacted. The walking controller drifts a little to the left (about 13 deg of heading per
    second), so a ball aimed at the fly's STARTING position would end up off to one side."""
    return cached_run("baseline_walk", force, azimuth=None, mode="no_brain")


def aimed_ball(azimuth: float, baseline: dict) -> LoomingBall:
    """A ball that leads its target: it flies at where the (non-reacting) fly's head will be
    when the ball arrives, from ``azimuth`` relative to where the fly will then be heading.
    Until the brain reacts the fly follows the baseline exactly, so at input onset the ball
    really is at that azimuth in the fly's frame (``summary`` reports the measured bearing)."""
    t_arrive = BALL["t_start"] + 1000.0 * (BALL["distance"] - BALL["stop_dist"]) / BALL["speed"]
    k = min(int(np.searchsorted(baseline["t"], t_arrive)), len(baseline["t"]) - 1)
    head, yaw = baseline["head"][k], float(baseline["yaw"][k])
    return LoomingBall.frontal(head, azimuth_deg=azimuth + yaw, **BALL)


def run(azimuth: float | None, encoder: str = "retinotopic", mode: str = "closed",
        seed: int = 0, steering: bool = True, walking: bool = True, render: bool = False,
        camera: str = "top", log_drive: bool = False, t_run: float = T_RUN,
        verbose: bool = False):
    """One closed-loop run with the ball coming from ``azimuth`` degrees (+ = left), or no ball
    (azimuth None). A walking fly gets a ball that leads it (``aimed_ball``); a standing fly a
    ball aimed at its head. Returns (log arrays, video frames, wall-clock seconds by stage)."""
    net = network(VARIANT)
    drive, groups = looming_drive(net.aligned)
    if azimuth is not None and walking:
        ball = aimed_ball(azimuth, baseline_walk())
    arena = FlyArena(ball_radius=p4.BALL_RADIUS, render=render, seed=seed, camera=camera)
    try:
        if azimuth is not None:
            arena.ball_path = (ball if walking
                               else LoomingBall.frontal(arena.head(), azimuth_deg=azimuth, **BALL))
        if mode == "no_brain":
            brain = None
            names = [f"{t} {s}" for t in ["DNp01", "DNp04", "DNp02", "DNp11"] for s in "LR"]
            if not log_drive:
                groups = []                    # nothing to drive
        else:
            brain, groups, names = p4.make_brain(VARIANT, seed=seed)
        enc = make_encoder(encoder, net.g.root_ids[drive])
        dec = (SteeringDecoder if steering else EscapeDecoder)(names, base=float(walking))
        loop = ClosedLoop(arena, brain, enc, dec, groups, mode=mode, log_drive=log_drive)
        log = loop.run(t_run, verbose=verbose).arrays()
        log["dn_names"] = np.array(names)
        return log, arena.frames(), loop.wall
    finally:
        arena.close()


def _path(name: str):
    return CACHE / f"{name}.npz"


def load_log(name: str) -> dict:
    with np.load(_path(name), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def cached_run(name: str, force: bool = False, gif=None, **kw) -> dict:
    """Load data/processed/phase5/<name>.npz, or run it (and optionally save a GIF) and cache."""
    if _path(name).exists() and not force:
        return load_log(name)
    log, frames, wall = run(**kw)
    log["wall_s"] = np.array([wall["vision"], wall["brain"], wall["body"]])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_path(name), **log)
    if gif is not None and frames:
        save_gif(frames, log, gif)
    return log


def run_name(encoder: str, azimuth: float, seed: int) -> str:
    return f"{encoder}_az{azimuth:+04.0f}_s{seed}"


def experiments(azimuths=AZIMUTHS, encoders=ENCODERS, seeds=SEEDS, force: bool = False,
                verbose: bool = True) -> dict:
    """Every closed-loop run of the notebook, cached. Returns {(encoder, azimuth): [logs]}."""
    out = {}
    for az in azimuths:
        for enc in encoders:
            out[(enc, az)] = []
            for s in seeds:
                if verbose and not _path(run_name(enc, az, s)).exists():
                    print(f"running {run_name(enc, az, s)}")
                out[(enc, az)].append(cached_run(run_name(enc, az, s), force, azimuth=az,
                                                 encoder=enc, seed=s))
    return out


RECORD_AZIMUTHS = (0, 30, 60, 90, 120, -90)


def record_drive(azimuths=RECORD_AZIMUTHS, encoders=ENCODERS, force: bool = False) -> dict:
    """Standing fly, no brain: the rate every one of the 314 input neurons receives while a ball
    approaches from each azimuth. Returns {(encoder, azimuth): log with a "drive" array}."""
    return {(enc, az): cached_run(f"record_{enc}_az{az:+04.0f}", force, azimuth=az,
                                  encoder=enc, mode="no_brain", walking=False, log_drive=True)
            for enc in encoders for az in azimuths}


def replay_brain(drive: np.ndarray, n_trials: int = 10, seed: int = 0,
                 tick_ms: float = 1.0) -> np.ndarray:
    """Feed a recorded per-neuron drive (ticks x 314, Hz) into the brain alone, n_trials
    independent Poisson trials at once. Returns escape-DN spike counts (n_DN, n_trials), DNs in
    the order of ``escape_names()``. No body, no feedback: a clean test of what the input
    pattern alone does to the DNs."""
    brain = _batched_brain(n_trials, seed)
    n_step = round(tick_ms / brain.params.dt)
    total = np.zeros((len(escape_names()), n_trials), dtype=np.int64)
    for row in np.asarray(drive, dtype=float):
        total += brain.step(n_step, rate_hz=np.repeat(row[:, None], n_trials, axis=1))
    return total


def escape_names() -> list[str]:
    return list(network(VARIANT).escape().keys())


def _batched_brain(n_trials: int, seed: int):
    """Phase 4's brain (LPLC2 + LC4 driven, the 8 escape DNs watched), n_trials columns wide."""
    import torch

    from c2c.sim.params import SHIU2024
    from c2c.sim.torch_lif import LIFStepper

    net = network(VARIANT)
    drive, _ = looming_drive(net.aligned)
    idx = list(net.escape().values())
    return LIFStepper(net.W, SHIU2024, drive=drive, n_trials=n_trials, seed=seed, watch=idx,
                      record=idx, device="cpu", dtype=torch.float32)


def replay_table(record: dict, n_trials: int = 10, force: bool = False) -> pd.DataFrame:
    """replay_brain for every recording; one row per (encoder, azimuth, DN, trial). Cached."""
    path = CACHE / "replay_counts.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    rows = []
    names = escape_names()
    for (enc, az), log in record.items():
        counts = replay_brain(log["drive"], n_trials=n_trials, seed=1)
        for j, n in enumerate(names):
            for trial in range(n_trials):
                rows.append({"encoder": enc, "azimuth": az, "dn": n.split()[0],
                             "dn_side": n.split()[1], "trial": trial,
                             "spikes": int(counts[j, trial])})
    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def gif_run(force: bool = False, azimuth: float = GIF_AZIMUTH, path=None) -> dict:
    """The README GIF: a ball from the side, retinotopic input, seen from above."""
    from c2c.paths import FIGURES

    path = path or FIGURES / f"phase5_side_approach_az{azimuth:+.0f}.gif"
    return cached_run(f"gif_az{azimuth:+04.0f}", force, gif=path, azimuth=azimuth,
                      encoder="retinotopic", seed=0, render=True, camera="top")


# ---- readouts -----------------------------------------------------------------------------------
def wrap(a):
    """Angles in degrees -> (-180, 180]."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def summary(log: dict, azimuth: float) -> dict:
    """What the fly did about a ball from ``azimuth``.

    All directions are in the fly's own frame AT INPUT ONSET (the first tick with any looming
    drive), so walking before the threat does not count: 0 = straight ahead, +90 = left.
      turn_deg        heading change from input onset to the end (+ = turned left)
      escape_dir_deg  direction of the displacement during the ESCAPE_WINDOW after onset
      bearing_deg     where the ball actually was at onset, in the same frame
      away            True if the fly turned away from the ball (turn and bearing have
                      opposite signs); undefined for a ball within 5 deg of straight ahead
    """
    t = log["t"]
    drive = log["lplc2"].sum(1) + log["lc4"].sum(1)
    has_input = bool(np.any(drive > 0))
    k0 = int(np.argmax(drive > 0)) if has_input else int(np.searchsorted(t, BALL["t_start"]))
    yaw = np.unwrap(np.radians(log["yaw"])) if len(log["yaw"]) else np.zeros(len(t))
    turn = float(np.degrees(yaw[-1] - yaw[k0]))
    psi0 = yaw[k0]
    k1 = min(int(np.searchsorted(t, t[k0] + ESCAPE_WINDOW)), len(t) - 1)
    d = log["thorax"][k1, :2] - log["thorax"][k0, :2]
    c, s = np.cos(-psi0), np.sin(-psi0)
    local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    cmd = log["command"]
    names = [str(n) for n in log["dn_names"]]
    counts = log["dn_counts"]
    out = {
        "azimuth": azimuth,
        "onset_ms": float(t[k0]) if has_input else np.nan,
        "turn_deg": turn,
        "escape_dir_deg": float(np.degrees(np.arctan2(local[1], local[0]))),
        "escape_dist_mm": float(np.hypot(*local)),
        "mean_steer": float(np.mean(cmd[k0:, 0] - cmd[k0:, 1]) / 2),
        "backed_up": bool(np.any(cmd[:, 0] + cmd[:, 1] < 0)),
        "min_gap_mm": float((np.linalg.norm(log["ball"] - log["head"], axis=1)
                             - p4.BALL_RADIUS).min()),
    }
    rel = log["ball"][k0, :2] - log["head"][k0, :2]           # where the ball was at onset
    out["bearing_deg"] = float(wrap(np.degrees(np.arctan2(rel[1], rel[0]) - psi0)))
    b = out["bearing_deg"]
    out["away"] = bool(np.sign(out["turn_deg"]) == -np.sign(b)) if abs(b) > 5 else np.nan
    for typ in ESCAPE_TYPES:
        for side in "LR":
            j = names.index(f"{typ} {side}")
            out[f"{typ} {side}"] = int(counts[:, j].sum())
    return out


def summary_table(runs: dict) -> pd.DataFrame:
    rows = []
    for (enc, az), logs in runs.items():
        for seed, log in enumerate(logs):
            rows.append({"encoder": enc, "seed": seed, **summary(log, az)})
    return pd.DataFrame(rows)


# ---- the connectome check: Dombrovski et al. 2023 -----------------------------------------------
def lc_dn_synapses(force: bool = False) -> pd.DataFrame:
    """One row per (LPLC2/LC4 neuron, escape DN): the neuron's RF position and how many synapses
    it makes onto that DN, every synapse counted (no threshold). ``ipsi`` is True when the DN
    is on the same side of the brain as the LC neuron."""
    from c2c.data import shiu
    from c2c.data.annotations import ANNOTATION_FILE, RAW

    path = CACHE / "lc_dn_synapses.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    rf, _ = receptive_field_table()
    ann = pd.read_csv(RAW / ANNOTATION_FILE, sep="\t", low_memory=False,
                      usecols=["root_id", "cell_type", "side"])
    dns = ann[ann["cell_type"].isin(ESCAPE_TYPES)]
    syn = shiu.output_synapses(rf["root_id"], dns["root_id"])
    grid = rf[["root_id", "cell_type", "side", "az", "el"]].merge(
        dns.rename(columns={"root_id": "dn_id", "cell_type": "dn", "side": "dn_side"}),
        how="cross")
    grid = grid.merge(syn.rename(columns={"pre": "root_id", "post": "dn_id"}),
                      on=["root_id", "dn_id"], how="left").fillna({"syn": 0})
    grid["syn"] = grid["syn"].astype(int)
    grid["ipsi"] = grid["side"] == grid["dn_side"]
    # Fold both sides onto one eye: |az| is "how far back" (0 front, 90 side) for either eye.
    grid["az_abs"] = grid["az"].abs()
    CACHE.mkdir(parents=True, exist_ok=True)
    grid.to_csv(path, index=False)
    return grid


def gradient(grid: pd.DataFrame, cell_type: str, dn: str, axis: str = "az_abs") -> dict:
    """Linear regression of synapse count (onto the IPSILATERAL ``dn``) against RF position.
    axis "az_abs": front -> back; "el": down -> up. Returns slope (synapses per 10 deg), r, p,
    n neurons, mean synapses."""
    from scipy.stats import linregress

    g = grid[(grid["cell_type"] == cell_type) & (grid["dn"] == dn) & grid["ipsi"]]
    g = g.groupby("root_id").agg(syn=("syn", "sum"), x=(axis, "first"))
    fit = linregress(g["x"], g["syn"])
    return {"cell_type": cell_type, "dn": dn, "axis": axis, "slope_per_10deg": 10 * fit.slope,
            "r": fit.rvalue, "p": fit.pvalue, "n": len(g), "mean_syn": g["syn"].mean()}


def ipsi_fraction(grid: pd.DataFrame) -> pd.DataFrame:
    """Share of each LC type's synapses onto each DN type that stay on the same side (S1)."""
    t = grid.groupby(["cell_type", "dn", "ipsi"])["syn"].sum().unstack("ipsi").fillna(0)
    t["ipsi_fraction"] = t[True] / (t[True] + t[False])
    return t.rename(columns={True: "ipsi_syn", False: "contra_syn"})


def drive_onto_dns(record: dict, grid: pd.DataFrame) -> pd.DataFrame:
    """Connectome-weighted input each escape DN type would get from the recorded retinotopic
    drive: sum over LC neurons of (mean rate during the approach) x (synapses onto that DN).
    A linear readout of the population code, no spiking involved: it isolates what the
    gradients alone do with the stimulus position."""
    net = network(VARIANT)
    drive, _ = looming_drive(net.aligned)
    ids = net.g.root_ids[drive]
    rows = []
    for az, log in record.items():
        rate = pd.Series(np.asarray(log["drive"]).mean(axis=0), index=ids)
        g = grid.assign(rate=grid["root_id"].map(rate).fillna(0.0))
        g["input"] = g["rate"] * g["syn"]
        for dn, v in g.groupby(["dn", "dn_side"])["input"].sum().items():
            rows.append({"azimuth": az, "dn": dn[0], "dn_side": dn[1], "input": v})
    return pd.DataFrame(rows)


# ---- steering calibration: system identification of the walking controller ---------------------
def steering_calibration(v_values=(1.0, 0.5, -0.5, -1.0),
                         w_values=(-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6),
                         t_settle: float = 200.0, t_measure: float = 800.0, seed: int = 0,
                         force: bool = False) -> pd.DataFrame:
    """Open-loop (no brain, no vision) yaw rate and speed for constant commands
    (s_L, s_R) = (v + w, v - w): the steering curve of the body. Cached."""
    path = CACHE / "steering_calibration.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    rows = []
    arena = FlyArena(seed=seed)
    n_settle, n_measure = round(t_settle / arena.dt_ms), round(t_measure / arena.dt_ms)
    try:
        for v in v_values:
            for w in w_values:
                arena.reset()
                cmd = np.clip([v + w, v - w], -1.2, 1.2)
                arena.step(cmd, n_settle)
                y0, p0 = arena.heading(), arena.thorax()
                yaws = []
                for _ in range(10):
                    arena.step(cmd, n_measure // 10)
                    yaws.append(arena.heading())
                dyaw = np.degrees(np.unwrap(np.radians([y0] + yaws)))[-1] - y0
                psi = np.radians(y0)
                d = arena.thorax()[:2] - p0[:2]
                fwd = np.cos(psi) * d[0] + np.sin(psi) * d[1]
                rows.append({"v": v, "w": w, "s_L": cmd[0], "s_R": cmd[1],
                             "yaw_rate_deg_s": dyaw / (t_measure / 1000.0),
                             "speed_mm_s": fwd / (t_measure / 1000.0)})
    finally:
        arena.close()
    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---- the GIF --------------------------------------------------------------------------------------
def save_gif(frames, log: dict, path, every: int = 2, width: int = 400) -> None:
    """Top-view frames -> an annotated GIF: time, left/right escape-DN rates, steering."""
    from PIL import Image, ImageDraw

    n, t = len(frames), log["t"]
    times = np.linspace(t[0], t[-1], n)
    names = [str(x) for x in log["dn_names"]]
    left = [i for i, x in enumerate(names) if x.endswith(" L") and not x.startswith("DNp01")]
    right = [i for i, x in enumerate(names) if x.endswith(" R") and not x.startswith("DNp01")]
    images = []
    for i in range(0, n, every):
        k = min(int(np.searchsorted(t, times[i])), len(t) - 1)
        img = Image.fromarray(frames[i]).convert("RGB")
        img = img.resize((width, round(img.height * width / img.width)))
        d = ImageDraw.Draw(img)
        sl, sr = log["command"][k]
        steer = (sl - sr) / 2
        label = "turning RIGHT" if steer > 0.05 else ("turning LEFT" if steer < -0.05 else "")
        if sl + sr < 0:
            label = ("backing up, " + label).strip(", ")
        rl, rr = log["dn_rates"][k][left].sum(), log["dn_rates"][k][right].sum()
        d.rectangle([0, 0, width, 36], fill=(255, 255, 255))
        d.text((6, 4), f"t = {times[i]:4.0f} ms   escape DNs  L {rl:4.0f} Hz | R {rr:4.0f} Hz",
               fill=(0, 0, 0))
        d.text((6, 20), f"legs  L {sl:+.2f}  R {sr:+.2f}   {label}", fill=(0, 0, 0))
        images.append(img)
    images[0].save(path, save_all=True, append_images=images[1:], duration=66, loop=0,
                   optimize=True)
