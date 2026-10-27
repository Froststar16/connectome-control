"""Tests for Phase 5: the eye map, receptive fields, the retinotopic encoder, the steering decoder.

Almost everything runs on small synthetic inputs whose answers are known in advance. Three tests
touch real things and skip themselves when those are missing:
  * the eye-map registration (needs the two eye-map CSVs in data/raw/),
  * the Dombrovski gradient check (needs the annotation and Shiu connectivity files),
  * the ommatidium geometry (needs flygym, renders the real eyes).
"""

import numpy as np
import pandas as pd
import pytest

from c2c.control.decoder import EscapeDecoder, SteeringDecoder
from c2c.control.encoder import RetinotopicEncoder, retinotopic_from_table
from c2c.control.loop import ClosedLoop
from c2c.data import eyemap as em
from c2c.paths import RAW

# ---- geometry helpers ---------------------------------------------------------------------------


def test_unit_vector_and_az_el_are_inverses():
    az = np.array([0.0, 45.0, -90.0, 170.0])
    el = np.array([0.0, 20.0, -45.0, 80.0])
    a, e = em.az_el(em.unit_vector(az, el))
    assert np.allclose(a, az) and np.allclose(e, el)
    # +90 azimuth is the fly's LEFT (+y), as in LoomingBall.frontal
    assert np.allclose(em.unit_vector(90.0, 0.0), [0, 1, 0], atol=1e-12)


def test_angle_between_unit_vectors():
    u = em.unit_vector(0.0, 0.0)
    assert em.angle_deg(u, em.unit_vector(30.0, 0.0)) == pytest.approx(30.0)
    assert em.angle_deg(u, em.unit_vector(0.0, -50.0)) == pytest.approx(50.0)


# ---- step A: grid registration ------------------------------------------------------------------
def hex_patch(radius=6):
    """A hexagonal patch of lattice sites as (p, q) integer pairs."""
    pq = [(p, q) for p in range(-radius, radius + 1) for q in range(-radius, radius + 1)
          if abs(p + q) <= radius]
    return np.array(pq)


def test_register_grid_recovers_a_known_shift():
    pq = hex_patch()
    columns = pd.DataFrame({"column_id": np.arange(len(pq)), "p": pq[:, 0], "q": pq[:, 1]})
    # The eye map is the same patch shifted by (dx, dy) = (3, 21) in lattice coordinates,
    # i.e. p += 12, q += 9 (since x = p - q, y = p + q).
    eye = pd.DataFrame({"p": pq[:, 0] + 12, "q": pq[:, 1] + 9, "x": 1.0, "y": 0.0, "z": 0.0})
    dx, dy, n = em.register_grid(columns, eye)
    assert (dx, dy) == (3, 21) and n == len(pq)


def test_column_directions_take_the_matching_eye_map_site():
    pq = hex_patch(2)
    columns = pd.DataFrame({"column_id": np.arange(len(pq)), "p": pq[:, 0], "q": pq[:, 1]})
    az = 10.0 * pq[:, 0]                                   # a direction unique to each site
    v = em.unit_vector(az, 0.0)
    eye = pd.DataFrame({"p": pq[:, 0], "q": pq[:, 1], "x": v[:, 0], "y": v[:, 1], "z": v[:, 2]})
    out = em.column_directions(columns, eye, shift=(0, 0))
    assert out["on_map"].all()
    assert np.allclose(out["az"].to_numpy(), az)


@pytest.mark.skipif(not (RAW / em.COLUMNS_FILE).exists() or not (RAW / em.EYEMAP_FILE).exists(),
                    reason="eye-map files not downloaded (scripts/download_data.py)")
def test_flywire_grid_registers_onto_the_eye_map():
    dx, dy, n = em.register_grid(em.load_columns(), em.load_eyemap())
    assert (dx, dy) == em.GRID_SHIFT and n >= 790          # 794 of 796 columns land on sites


# ---- step B: mirror fit -------------------------------------------------------------------------
def test_fit_mirror_recovers_a_reflection():
    """Right cells on a jittered grid; left cells are their exact mirror image about x = 500 um,
    shifted a little. The fitted map must put every left cell back on its right partner."""
    rng = np.random.default_rng(0)
    n = 400
    right_um = np.c_[rng.uniform(600, 700, n), rng.uniform(100, 200, n), rng.uniform(50, 150, n)]
    left_um = right_um * [-1, 1, 1] + [1000 + 7.0, 3.0, -2.0]
    types = np.where(np.arange(n) % 2, "T4a", "Mi1")
    vox = 1.0 / em.UM_PER_VOXEL
    ann = pd.DataFrame({
        "root_id": np.arange(2 * n), "cell_type": np.r_[types, types],
        "side": ["right"] * n + ["left"] * n,
        "soma_x": np.r_[right_um[:, 0], left_um[:, 0]] * vox[0],
        "soma_y": np.r_[right_um[:, 1], left_um[:, 1]] * vox[1],
        "soma_z": np.r_[right_um[:, 2], left_um[:, 2]] * vox[2]})
    columns = pd.DataFrame({"root_id": np.arange(n), "type": types, "column_id": np.arange(n),
                            "p": 0, "q": 0})
    A, b, residual = em.fit_mirror(ann, columns, n_iter=20)
    assert np.linalg.det(A) == pytest.approx(-1.0, abs=1e-3)       # a reflection
    assert np.allclose(left_um @ A.T + b, right_um, atol=0.5)
    assert residual < 0.5


# ---- step C: receptive fields ------------------------------------------------------------------
def test_receptive_field_is_the_synapse_weighted_input_direction():
    """Target 1 (left) gets 3 synapses from a cell looking at az 30 and 1 from one at az 70, plus
    input from a right-side cell that must be ignored (wrong side) and from an unmapped cell."""
    d30, d70 = em.unit_vector(30.0, 0.0), em.unit_vector(70.0, 0.0)
    cells = pd.DataFrame({"root_id": [10, 11, 12], "side": ["left", "left", "right"],
                          "x": [d30[0], d70[0], 1.0], "y": [d30[1], d70[1], 0.0],
                          "z": [0.0, 0.0, 0.0]})
    inputs = pd.DataFrame({"pre": [10, 11, 12, 99], "post": [1, 1, 1, 1], "syn": [3, 1, 5, 2]})
    targets = pd.DataFrame({"root_id": [1], "cell_type": ["LC4"], "side": ["left"]})
    rf = em.receptive_fields(inputs, cells, targets).iloc[0]
    centre = 3 * d30 + d70
    centre /= np.linalg.norm(centre)
    assert rf["az"] == pytest.approx(em.az_el(centre)[0])
    assert rf["syn_mapped"] == 4 and rf["frac_mapped"] == pytest.approx(4 / 11)
    a30, a70 = em.angle_deg(d30, centre), em.angle_deg(d70, centre)
    assert rf["sigma"] == pytest.approx(np.sqrt((3 * a30**2 + a70**2) / 4 / 2))


# ---- the retinotopic encoder --------------------------------------------------------------------
def eye_grid():
    """A toy compound eye: ommatidia every 5 deg in azimuth (-10 ... 140) and elevation
    (-40 ... 40), the right eye the mirror image of the left. (2, n, 3) unit vectors."""
    az, el = np.meshgrid(np.arange(-10, 141, 5.0), np.arange(-40, 41, 5.0))
    left = em.unit_vector(az.ravel(), el.ravel())
    return np.stack([left, left * [1, -1, 1]]), az.ravel(), el.ravel()


def readout(dark):
    """(2, n) dark flags (1 = black ommatidium) -> flygym-style (2, n, 2) readout, sky = 1."""
    b = np.where(np.asarray(dark) > 0, 0.0, 1.0)
    out = np.zeros(b.shape + (2,))
    out[..., 0] = b
    return out


def toy_encoder(**kw):
    omm, _, _ = eye_grid()
    # neuron 0: left eye, looking at az 20 / el 0; neuron 1: left eye, az 100; neuron 2: right eye
    # az -20 (mirror of neuron 0); all LC4, sigma 10 deg
    dirs = em.unit_vector(np.array([20.0, 100.0, -20.0]), np.zeros(3))
    return RetinotopicEncoder(rf_dirs=dirs, rf_sigma=np.full(3, 10.0), eye=np.array([0, 0, 1]),
                              cell_type=np.array(["LC4"] * 3), omm_dirs=omm, **kw)


def test_spatial_factor_is_one_at_the_centre_and_gaussian_away_from_it():
    enc = toy_encoder()
    _, az, el = eye_grid()
    for offset in [0.0, 10.0, 20.0]:
        dark = np.zeros((2, len(az)))
        dark[0, (az == 20.0 + offset) & (el == 0.0)] = 1           # one black ommatidium
        h = enc.spatial(readout(dark))
        assert h[0] == pytest.approx(np.exp(-offset**2 / (2 * 10.0**2)))
        assert h[2] == 0.0                                         # the other eye sees nothing


def test_nothing_dark_means_no_spatial_drive():
    enc = toy_encoder()
    _, az, _ = eye_grid()
    assert np.all(enc.spatial(readout(np.zeros((2, len(az))))) == 0.0)


def test_sigma_scale_widens_the_receptive_field():
    narrow, wide = toy_encoder(), toy_encoder(sigma_scale=2.0)
    _, az, el = eye_grid()
    dark = np.zeros((2, len(az)))
    dark[0, (az == 40.0) & (el == 0.0)] = 1                        # 20 deg from neuron 0
    assert wide.spatial(readout(dark))[0] == pytest.approx(np.exp(-20.0**2 / (2 * 20.0**2)))
    assert narrow.spatial(readout(dark))[0] < wide.spatial(readout(dark))[0]


def test_object_position_is_coded_by_which_neurons_fire():
    """A growing object at az 100 (left eye) drives neuron 1 and not neurons 0 or 2; the same
    object at az -20 (right eye) drives neuron 2 only. Same features, different population."""
    _, az, el = eye_grid()
    for centre, expect in [(100.0, 1), (-20.0, 2)]:
        enc = toy_encoder()
        eye = 0 if centre > 0 else 1
        for k, t in enumerate(np.arange(0.0, 200.0, 2.0)):
            radius = 2.0 + 0.15 * t                                # expanding disk, deg
            dark = np.zeros((2, len(az)))
            dark[eye] = np.hypot(az - abs(centre), el) <= radius
            rates = enc.update(readout(dark), t)
        r = rates["neurons"]
        assert r[expect] > 0
        assert np.all(r[np.arange(3) != expect] < 1e-3 * r[expect])


def test_covered_receptive_fields_give_the_phase4_rate():
    """Two disks grow over all three RF centres (left eye az 20 and az 100, right eye az -20).
    Once every centre is covered, h = 1 and each neuron gets exactly the rate the uniform
    Phase 4 encoder gives that eye: the retinotopic encoder reduces to the uniform one."""
    from c2c.control.encoder import LoomingEncoder

    retino, uniform = toy_encoder(), LoomingEncoder()
    _, az, el = eye_grid()
    for t in np.arange(0.0, 240.0, 2.0):
        r = 3.0 + 0.1 * t                                          # disk radius, deg
        disk = (np.hypot(az - 20, el) <= r) | (np.hypot(az - 100, el) <= r)
        frame = readout(np.stack([disk, disk]))                    # same pattern in both eyes
        rr, ru = retino.update(frame, t), uniform.update(frame, t)
    assert np.allclose(retino.h, 1.0)
    assert ru["LC4"][0] > 0
    assert np.allclose(rr["neurons"], [ru["LC4"][0], ru["LC4"][0], ru["LC4"][1]])


def test_retinotopic_from_table_follows_drive_order():
    rf = pd.DataFrame({"root_id": [5, 6, 7], "cell_type": ["LC4", "LPLC2", "LC4"],
                       "side": ["right", "left", "left"], "sigma": [8.0, 9.0, 10.0]})
    rf[["x", "y", "z"]] = em.unit_vector(np.array([-30.0, 40.0, 80.0]), np.zeros(3))
    omm, _, _ = eye_grid()
    enc = retinotopic_from_table(rf, [6, 7, 5], omm)
    assert enc.cell_type.tolist() == ["LPLC2", "LC4", "LC4"]
    assert enc.eye.tolist() == [0, 0, 1]
    assert enc.rf_sigma.tolist() == [9.0, 10.0, 8.0]
    rf.loc[0, "sigma"] = np.nan
    with pytest.raises(ValueError):
        retinotopic_from_table(rf, [6, 7, 5], omm)


# ---- the steering decoder -----------------------------------------------------------------------
NAMES = [f"{t} {s}" for t in ["DNp01", "DNp04", "DNp02", "DNp11"] for s in "LR"]


def rates(**hz):
    """Rates in NAMES order from keyword arguments like DNp02_L=40."""
    return np.array([hz.get(n.replace(" ", "_"), 0.0) for n in NAMES], dtype=float)


def test_symmetric_activity_gives_the_phase4_command_on_both_sides():
    steer, plain = SteeringDecoder(NAMES), EscapeDecoder(NAMES)
    for r in [rates(), rates(DNp02_L=30, DNp02_R=30, DNp04_L=10, DNp04_R=10),
              rates(DNp11_L=80, DNp11_R=80)]:
        s = steer.command(r)
        assert s[0] == pytest.approx(s[1])
        assert s[0] == pytest.approx(np.clip(plain.command(r)[0], -1.2, 1.2))


def test_escape_drive_on_the_left_turns_right():
    dec = SteeringDecoder(NAMES)
    s = dec.command(rates(DNp02_L=40, DNp04_L=40))       # D_L = 80/40 = 2, D_R = 0
    assert s[0] > s[1]                                   # left legs harder -> clockwise
    assert dec.turn(rates(DNp02_L=40, DNp04_L=40)) == pytest.approx(dec.k_turn * 2 / 3)


def test_steering_is_mirror_symmetric():
    dec = SteeringDecoder(NAMES)
    left = dec.command(rates(DNp11_L=60, DNp02_L=20, DNp02_R=5))
    right = dec.command(rates(DNp11_R=60, DNp02_R=20, DNp02_L=5))
    assert np.allclose(left, right[::-1])


def test_giant_fiber_does_not_steer():
    dec = SteeringDecoder(NAMES)
    assert dec.turn(rates(DNp01_L=300)) == 0.0


def test_steering_command_stays_in_range():
    dec = SteeringDecoder(NAMES)
    rng = np.random.default_rng(1)
    for _ in range(1000):
        s = dec.command(rng.exponential(80.0, len(NAMES)))
        assert np.all(np.abs(s) <= dec.v_max + 1e-12)


# ---- the loop with per-neuron input -------------------------------------------------------------
class FakeArena:
    dt_ms = 0.1

    def __init__(self, ommatidia):
        self.k, self.omm = 0, ommatidia

    @property
    def t_ms(self):
        return self.k * self.dt_ms

    def vision(self):
        return self.omm

    def step(self, command, n):
        self.k += n

    def thorax(self):
        return np.zeros(3)

    head = ball = thorax

    def heading(self):
        return 0.0


class FakeBrain:
    class params:
        dt = 0.1

    def __init__(self):
        self.rates = []

    def step(self, n, rate_hz):
        self.rates.append(np.asarray(rate_hz).copy())
        return np.zeros((len(NAMES), 1), dtype=np.int64)


def test_loop_gives_the_brain_one_rate_per_neuron():
    enc = toy_encoder()
    _, az, el = eye_grid()
    dark = np.zeros((2, len(az)))
    dark[0, np.hypot(az - 100, el) <= 15] = 1
    brain = FakeBrain()
    groups = [("LC4", "left", 2), ("LC4", "right", 1)]
    loop = ClosedLoop(FakeArena(readout(dark)), brain, enc, SteeringDecoder(NAMES), groups,
                      log_drive=True)
    log = loop.run(5.0).arrays()
    assert brain.rates[0].shape == (3, 1)
    assert log["drive"].shape == (5, 3) and log["yaw"].shape == (5,)
    assert np.allclose(brain.rates[-1][:, 0], log["drive"][-1])


# ---- real data and the real body (skipped if missing) -------------------------------------------
HAVE_CONNECTOME = all((RAW / f).exists() for f in [
    "neuron_annotations_v3.2.0.tsv", "shiu2024_connectivity_783.parquet", em.COLUMNS_FILE,
    em.EYEMAP_FILE])


@pytest.mark.skipif(not HAVE_CONNECTOME, reason="connectome / eye-map files not downloaded")
def test_lc4_gradients_match_dombrovski_2023(tmp_path, monkeypatch):
    """The prediction: LC4 neurons looking FORWARD synapse more onto DNp02 (backward escape),
    those looking BACK more onto DNp11 (forward escape); dorsal LPLC2 more onto the giant
    fiber. Recomputed here from our own receptive fields."""
    from c2c.analysis import phase5 as p5

    monkeypatch.setattr(p5, "CACHE", tmp_path)
    grid = p5.lc_dn_synapses()
    assert grid.loc[~grid["ipsi"], "syn"].sum() == 0          # all LC -> DN synapses ipsilateral
    back = p5.gradient(grid, "LC4", "DNp02", "az_abs")
    fwd = p5.gradient(grid, "LC4", "DNp11", "az_abs")
    gf = p5.gradient(grid, "LPLC2", "DNp01", "el")
    assert back["slope_per_10deg"] < 0 and back["p"] < 1e-3
    assert fwd["slope_per_10deg"] > 0 and fwd["p"] < 1e-3
    assert gf["slope_per_10deg"] > 0 and gf["p"] < 1e-3


def test_ommatidium_directions_match_the_rendered_eyes():
    """Put a small ball along the computed direction of an ommatidium: the dark patch in the
    rendered eye must be centred on that direction (within 3 deg)."""
    pytest.importorskip("flygym")
    from c2c.control.body import FlyArena

    arena = FlyArena(ball_radius=1.5)
    try:
        dirs = arena.ommatidia_directions()
        mj_data = arena.sim.mj_data
        cams = arena.sim._intern_eye_camera_ids_by_fly[arena.fly.name]
        enc = toy_encoder()
        for eye, k in [(0, 468), (1, 525), (0, 54)]:          # above the horizon, both eyes
            mj_data.mocap_pos[0] = mj_data.cam_xpos[cams[eye]] + 20.0 * dirs[eye, k]
            arena.sim.forward()
            c = enc.covered(arena.vision())[eye]
            assert c.sum() > 0
            centre = (c[:, None] * dirs[eye]).sum(0)
            centre /= np.linalg.norm(centre)
            assert em.angle_deg(centre, dirs[eye, k]) < 3.0
    finally:
        arena.close()


# ---- experiment helpers ---------------------------------------------------------------------------
def test_aimed_ball_leads_the_walking_fly():
    """The baseline fly walks along +x at 10 mm/s, heading drifting to 30 deg. The ball must
    end up at the head position at arrival time, coming from the requested azimuth relative to
    the heading the fly will have then."""
    from c2c.analysis import phase5 as p5

    t = np.arange(0.0, 1000.0, 1.0)
    head = np.c_[0.01 * t, np.zeros_like(t), np.full_like(t, 1.0)]
    base = {"t": t, "head": head, "yaw": 0.03 * t}
    ball = p5.aimed_ball(45.0, base)
    t_arrive = p5.BALL["t_start"] + 1000 * (p5.BALL["distance"] - p5.BALL["stop_dist"]) \
        / p5.BALL["speed"]
    k = int(np.searchsorted(t, t_arrive))
    assert np.allclose(ball.target, head[k])
    approach_az = em.az_el(ball.start - ball.target)[0]
    assert approach_az == pytest.approx(45.0 + 0.03 * t[k])


def test_summary_reads_turn_direction_and_bearing():
    """A synthetic run: input starts at t = 100 ms with the ball 90 deg to the left; the fly then
    turns 40 deg to the right while walking forward."""
    from c2c.analysis import phase5 as p5

    n = 600
    t = np.arange(n, dtype=float)
    yaw = np.where(t < 100, 0.0, -40.0 * np.clip((t - 100) / 300, 0, 1))
    psi = np.radians(yaw)
    pos = np.cumsum(0.01 * np.c_[np.cos(psi), np.sin(psi), np.zeros(n)], axis=0)
    drive = (t >= 100).astype(float)
    log = {"t": t, "lplc2": np.c_[drive, 0 * drive], "lc4": np.c_[drive, 0 * drive],
           "yaw": yaw, "thorax": pos, "head": pos, "ball": pos + [0.0, 20.0, 0.0],
           "command": np.c_[np.full(n, 1.2), np.full(n, 0.8)], "dn_names": np.array(NAMES),
           "dn_counts": np.zeros((n, len(NAMES)), dtype=int)}
    s = p5.summary(log, 90.0)
    assert s["onset_ms"] == 100.0
    assert s["bearing_deg"] == pytest.approx(90.0, abs=1.0)
    assert s["turn_deg"] == pytest.approx(-40.0)
    assert s["away"] is True
    assert -40.0 < s["escape_dir_deg"] < 0.0            # forward and to the right
    assert s["mean_steer"] == pytest.approx(0.2)
