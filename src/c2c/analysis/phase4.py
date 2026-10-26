"""The Phase 4 experiments: the spiking brain driving a simulated fly body.

Each run is cached to data/processed/phase4/<name>.npz (one row per 1 ms control tick, see
c2c.control.loop.LoopLog), so the notebook can reload them instantly. ``scripts/run_phase4.py``
runs them all from a terminal; delete a cache file (or pass force=True) to re-run one.

The stimulus is the same in every run: a black ball 2 mm in radius starts 60 mm in front of the
fly, 20 deg above the horizon, and from t = 100 ms flies straight at the fly's starting head
position at 150 mm/s (r/v = 13 ms, in the range of looming stimuli used on flies). It stops 8 mm
short of that point at ~450 ms and stays there, so the second half of each run asks a different
question: what does the fly do next to an object that is no longer moving?

Runs
----
record       no brain, fly standing still, eyes open: what the looming detectors would report to
             a stationary fly. This trace is the "open-loop stimulus".
no_brain     the fly walks forward, brain disconnected: the baseline, it walks into the ball.
open         the recorded trace is replayed into the brain while the body walks. The body reacts,
             but its movement cannot change what the brain receives.
closed       the brain sees through the live eyes. Its own escape changes its input.
closed_-LC4, closed_-LPLC2
             closed loop with one input type silenced (outgoing synapses removed, as in Phase 3).
"""

from __future__ import annotations

import numpy as np

from c2c.analysis.phase3 import network
from c2c.control.body import FlyArena, LoomingBall
from c2c.control.decoder import EscapeDecoder
from c2c.control.encoder import LoomingEncoder, looming_drive
from c2c.control.loop import ClosedLoop, replay_from_log
from c2c.data.annotations import select
from c2c.paths import PROCESSED
from c2c.sim.network import silence_columns
from c2c.sim.params import SHIU2024

CACHE = PROCESSED / "phase4"
T_RUN = 1000.0         # ms per run
LOOM_ONSET = 100.0     # ms, ball starts moving
BALL = {"distance": 60.0, "elevation_deg": 20.0, "speed": 150.0, "t_start": LOOM_ONSET,
        "stop_dist": 8.0}
BALL_RADIUS = 2.0
STANDOFF_FROM = 600.0  # ms; the ball stops at ~450 ms, "standoff" = what the fly does after
SILENCE = {"intact": (), "-LC4": ("LC4",), "-LPLC2": ("LPLC2",)}


def make_brain(variant: str, silence=(), seed: int = 0):
    """LIFStepper for one fly: LPLC2 + LC4 as Poisson-driven inputs, the 8 escape DNs watched.
    Returns (brain, drive_groups, dn_names)."""
    import torch

    from c2c.sim.torch_lif import LIFStepper

    net = network(variant)
    drive, groups = looming_drive(net.aligned)
    W = net.W
    if silence:
        W = silence_columns(W, select(net.aligned, cell_type=list(silence)))
    dns = net.escape()                          # {"DNp01 L": index, ...}
    idx = list(dns.values())
    brain = LIFStepper(W, SHIU2024, drive=drive, n_trials=1, seed=seed, watch=idx, record=idx,
                       device="cpu", dtype=torch.float32)
    return brain, groups, list(dns.keys())


def run(mode: str, variant: str = "shiu_min5", silence=(), seed: int = 0, render: bool = False,
        replay=None, walking: bool = True, t_run: float = T_RUN, verbose: bool = False):
    """One run of the loop. Returns (log arrays, video frames, wall-clock seconds by stage)."""
    arena = FlyArena(ball_radius=BALL_RADIUS, render=render, seed=seed)
    try:
        arena.ball_path = LoomingBall.frontal(arena.head(), **BALL)
        if mode == "no_brain":
            brain, groups = None, []
            names = [f"{t} {s}" for t in ["DNp01", "DNp04", "DNp02", "DNp11"] for s in "LR"]
        else:
            brain, groups, names = make_brain(variant, silence, seed)
        decoder = EscapeDecoder(names, base=1.0 if walking else 0.0)
        loop = ClosedLoop(arena, brain, LoomingEncoder(), decoder, groups, mode=mode,
                          replay=replay)
        log = loop.run(t_run, verbose=verbose).arrays()
        log["dn_names"] = np.array(names)
        frames = arena.frames()
        return log, frames, loop.wall
    finally:
        arena.close()


def _path(name: str):
    return CACHE / f"{name}.npz"


def save_log(log: dict, name: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_path(name), **log)


def load_log(name: str) -> dict:
    with np.load(_path(name), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def cached_run(name: str, force: bool = False, gif=None, **kw) -> dict:
    """Load data/processed/phase4/<name>.npz, or run it (and optionally save a GIF) and cache."""
    if _path(name).exists() and not force:
        return load_log(name)
    log, frames, wall = run(**kw)
    log["wall_s"] = np.array([wall["vision"], wall["brain"], wall["body"]])
    save_log(log, name)
    if gif is not None and frames:
        save_gif(frames, log, gif)
    return log


# ---- the experiments ---------------------------------------------------------------------------
def record_stimulus(force: bool = False) -> dict:
    return cached_run("record", force, mode="no_brain", walking=False)


def experiments(variant: str = "shiu_min5", seeds=(0, 1, 2), force: bool = False, verbose=True,
                gif=None) -> dict[str, list[dict]]:
    """Every run of the notebook, cached. Returns {condition: [log per seed]}."""
    out = {"record": [record_stimulus(force)]}
    replay = replay_from_log(out["record"][0])
    out["no_brain"] = [cached_run("no_brain", force, mode="no_brain")]
    out["open"] = [cached_run(f"open_{variant}_s{s}", force, mode="open", variant=variant,
                              replay=replay, seed=s, verbose=verbose) for s in seeds]
    for cond, sil in SILENCE.items():
        key = "closed" if cond == "intact" else f"closed_{cond}"
        out[key] = [cached_run(f"{key}_{variant}_s{s}", force, mode="closed", variant=variant,
                               silence=sil, seed=s, verbose=verbose,
                               render=(gif is not None and cond == "intact" and s == seeds[0]),
                               gif=gif if (cond == "intact" and s == seeds[0]) else None)
                    for s in seeds]
    return out


# ---- which connectome? the giant-fiber question from Phase 3 ----------------------------------
FACTORS = {   # variant -> (edge threshold, sign rule, modulatory edges)
    "ours": (">= 5", "per edge", "0"),
    "ours_mod": (">= 5", "per edge", "+1"),
    "shiu_min5": (">= 5", "per neuron", "+1"),
    "shiu_full": ("all", "per neuron", "+1"),
}


def giant_fiber_factorial(variants=tuple(FACTORS), input_hz: float = 100.0):
    """Phase 3's looming sweep on each graph variant, read out at one input rate.

    Neighbouring variants differ in ONE choice: ours -> ours_mod (modulatory sign),
    ours_mod -> shiu_min5 (per-edge vs per-neuron signs, with modulators excitatory in both),
    shiu_min5 -> shiu_full (synapse threshold). So the row where the giant fiber starts firing
    names the cause. Uses (and caches) c2c.analysis.phase3.looming_sweep."""
    import pandas as pd

    from c2c.analysis import phase3 as p3
    from c2c.analysis import spiking as sp

    rows = []
    for v in variants:
        try:
            res, rates = p3.looming_sweep(v)
        except FileNotFoundError as e:       # e.g. ours without the Zenodo connections table
            print(f"skipping {v}: {e}")
            continue
        tab = sp.sweep_table(res, rates, network(v).escape())
        tab = tab[tab.input_hz == input_hz]
        for _, r in tab.iterrows():
            rows.append({"variant": v, "threshold": FACTORS[v][0], "signs": FACTORS[v][1],
                         "modulatory": FACTORS[v][2], "neuron": r["neuron"],
                         "rate_hz": r["rate_hz"], "p_fire": r["p_fire"],
                         "latency_ms": r["latency_ms"]})
    return pd.DataFrame(rows)


# ---- readouts ----------------------------------------------------------------------------------
def summary(log: dict) -> dict:
    """The numbers that say whether (and how) the fly escaped."""
    t = log["t"]
    head, ball = log["head"], log["ball"]
    dist = np.linalg.norm(ball - head, axis=1) - BALL_RADIUS        # gap between head and ball
    x = log["thorax"][:, 0] - log["thorax"][0, 0]
    drive = np.r_[log["lplc2"].sum(1) + log["lc4"].sum(1)]
    onset = t[np.argmax(drive > 0)] if np.any(drive > 0) else np.nan
    cmd = log["command"][:, 0]
    back = t[np.argmax(cmd < 0)] if np.any(cmd < 0) else np.nan
    late = t >= STANDOFF_FROM
    out = {
        "input_onset_ms": onset,
        "backward_cmd_ms": back,                      # first tick the body was told to back up
        "cmd_latency_ms": back - onset,
        "min_gap_mm": float(dist.min()),
        "standoff_gap_mm": float(dist[late].mean()),  # mean gap once the ball has stopped
        "x_at_loom_mm": float(np.interp(LOOM_ONSET, t, x)),
        "x_final_mm": float(x[-1]),
        "x_max_mm": float(x.max()),
        "takeoff_ms": t[np.argmax(log["takeoff"])] if log["takeoff"].any() else np.nan,
    }
    names = [str(n) for n in log.get("dn_names", [])]
    counts = log["dn_counts"]
    for j, n in enumerate(names):
        spikes = np.flatnonzero(counts[:, j])
        out[f"first {n}"] = (t[spikes[0]] + 1.0 - onset) if len(spikes) else np.nan
        out[f"spikes {n}"] = int(counts[:, j].sum())
    return out


def save_gif(frames, log: dict, path, every: int = 2, width: int = 480) -> None:
    """Rendered side-view frames -> an annotated GIF for the README."""
    from PIL import Image, ImageDraw

    n = len(frames)
    t = log["t"]
    times = np.linspace(t[0], t[-1], n)                 # renderer frames are evenly spaced
    images = []
    for i in range(0, n, every):
        k = int(np.searchsorted(t, times[i]))
        k = min(k, len(t) - 1)
        img = Image.fromarray(frames[i]).convert("RGB")
        img = img.resize((width, round(img.height * width / img.width)))
        d = ImageDraw.Draw(img)
        cmd = float(log["command"][k, 0])
        rates = log["dn_rates"][k]
        label = "BACKING UP" if cmd < 0 else "walking"
        d.rectangle([0, 0, width, 22], fill=(255, 255, 255))
        d.text((6, 5), f"t = {times[i]:5.0f} ms   escape DNs {rates.mean():5.0f} Hz   "
                       f"command {cmd:+.2f}  {label}", fill=(0, 0, 0))
        images.append(img)
    images[0].save(path, save_all=True, append_images=images[1:], duration=66, loop=0,
                   optimize=True)
