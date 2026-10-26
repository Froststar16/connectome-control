"""Tests for the Phase 4 encoder, decoder and loop, on inputs whose answers are known.

Most tests build synthetic eye images or spike counts in numpy, so they run without flygym or a
GPU. The calibration test at the end renders the real NeuroMechFly eyes and is skipped if flygym
is not installed.
"""

import numpy as np
import pandas as pd
import pytest

from c2c.control.decoder import EscapeDecoder
from c2c.control.encoder import LoomingEncoder, looming_drive, rates_for_drive
from c2c.control.loop import ClosedLoop, replay_from_log

N_OMM = 721
SKY, FLOOR = 1.0, 0.15


def eyes(dark_left: float = 0.0, dark_right: float = 0.0, background: float = SKY) -> np.ndarray:
    """Synthetic flygym readout (2, 721, 2): the first k ommatidia of each eye are black.
    A fractional k makes the next ommatidium partly dark. Ommatidia alternate yellow/pale, with
    the other channel 0, like the real readout."""
    out = np.zeros((2, N_OMM, 2), dtype=np.float32)
    for e, k in enumerate([dark_left, dark_right]):
        b = np.full(N_OMM, background)
        n = int(k)
        b[:n] = 0.0
        if k > n:  # partial coverage: brightness in between
            enc = LoomingEncoder()
            b[n] = enc.dark_hi - (k - n) * (enc.dark_hi - enc.dark_lo)
        kind = np.arange(N_OMM) % 2
        out[e, np.arange(N_OMM), kind] = b
    return out


def ball_area(t_ms, d0=60.0, speed=150.0, radius=2.0, dphi=5.5):
    """Dark area (ommatidia) of a ball approaching at constant speed, from the encoder's own
    size model: theta = 2 asin(r / d), area = (theta / dphi)^2."""
    d = np.maximum(d0 - speed * np.asarray(t_ms) / 1000.0, radius * 1.05)
    theta = 2 * np.degrees(np.arcsin(radius / d))
    return (theta / dphi) ** 2


# ---- encoder ----------------------------------------------------------------------------------
def test_dark_area_counts_black_ommatidia_only():
    enc = LoomingEncoder()
    assert np.allclose(enc.dark_area(eyes(0, 0)), [0, 0])
    assert np.allclose(enc.dark_area(eyes(0, 0, background=FLOOR)), [0, 0])  # floor is not dark
    assert np.allclose(enc.dark_area(eyes(12, 3)), [12, 3])
    assert np.allclose(enc.dark_area(eyes(12.5, 0)), [12.5, 0], atol=1e-6)  # soft edge


def test_angular_size_inverts_the_area_model():
    enc = LoomingEncoder()
    for theta in [11.0, 25.0, 60.0]:
        area = (theta / enc.dphi_deg) ** 2
        assert enc.angular_size(area) == pytest.approx(theta)


def run_encoder(area_fn, t_end=400.0, frame_ms=2.0, sides=(1.0, 1.0)):
    enc = LoomingEncoder()
    rows = []
    for t in np.arange(0.0, t_end, frame_ms):
        a = area_fn(t)
        r = enc.update(eyes(a * sides[0], a * sides[1]), t)
        rows.append((t, enc.theta, enc.velocity, *r["LPLC2"], *r["LC4"]))
    return np.array(rows)


def test_approaching_ball_drives_lc4_then_lplc2():
    """Constant-speed approach: expansion speed and size both grow, so both rates rise, LC4
    (speed) responds and saturates before LPLC2 (size) does."""
    rows = run_encoder(ball_area)
    t, theta, _, lp_l, lp_r, lc_l, lc_r = rows.T
    assert lc_l[t < 100].max() == 0 and lp_l[t < 100].max() == 0      # far away: nothing
    assert lc_l.max() == pytest.approx(150.0) and lp_l.max() > 50.0
    assert t[np.argmax(lc_l > 75)] < t[np.argmax(lp_l > 75)]            # speed before size
    assert np.array_equal(lc_l, lc_r) and np.array_equal(lp_l, lp_r)    # frontal: both sides
    true = 2 * np.degrees(np.arcsin(2.0 / np.maximum(60 - 0.15 * t, 2.1)))
    late = (t > 250) & (t < 340)
    # the low-pass makes the estimate lag; it should stay within ~25% of the truth
    assert np.all(np.abs(theta[late] - true[late]) / true[late] < 0.25)


def test_static_or_receding_objects_are_not_looming():
    big = (30.0 / 5.5) ** 2                                         # a 30 deg object
    static = run_encoder(lambda t: big)
    assert static[50:, 3:].max() == 0                               # no rates once settled
    receding = run_encoder(lambda t: ball_area(400.0 - t))
    assert receding[50:, 3:].max() == 0


def test_object_in_one_eye_drives_that_side_only():
    rows = run_encoder(ball_area, sides=(1.0, 0.0))
    _, _, _, lp_l, lp_r, lc_l, lc_r = rows.T
    assert lc_l.max() > 0 and lp_l.max() > 0
    assert lc_r.max() == 0 and lp_r.max() == 0


def test_rates_for_drive_follows_group_order():
    aligned = pd.DataFrame({"cell_type": ["LC4", "LPLC2", "LC4", "X", "LPLC2", "LC4"],
                            "side": ["left", "left", "right", "left", "right", "left"]})
    drive, groups = looming_drive(aligned)
    assert drive.tolist() == [1, 4, 0, 5, 2]                    # LPLC2 L, R; LC4 L, L, R
    assert groups == [("LPLC2", "left", 1), ("LPLC2", "right", 1),
                      ("LC4", "left", 2), ("LC4", "right", 1)]
    r = rates_for_drive(groups, {"LPLC2": np.array([10.0, 20.0]),
                                 "LC4": np.array([30.0, 40.0])})
    assert r.tolist() == [10, 20, 30, 30, 40]


# ---- decoder ----------------------------------------------------------------------------------
NAMES = [f"{t} {s}" for t in ["DNp01", "DNp04", "DNp02", "DNp11"] for s in "LR"]


def counts(**hz_per_type):
    """Spike counts for a 1 ms tick, as expected values (fractions are fine for the filter)."""
    c = np.zeros(len(NAMES))
    for i, n in enumerate(NAMES):
        c[i] = hz_per_type.get(n.split(" ")[0], 0.0) / 1000.0
    return c


def test_rate_filter_converges_to_the_true_rate():
    dec = EscapeDecoder(NAMES)
    train = np.zeros((1000, len(NAMES)))
    train[::10, :] = 1                                   # one spike every 10 ms = 100 Hz
    for c in train:
        dec.update_rates(c, 1.0)
    assert np.allclose(dec.rates, 100.0, rtol=0.4)      # last spike timing makes it ripple
    mean = np.mean([dec.update_rates(c, 1.0)[0] for c in train[:500]])
    assert mean == pytest.approx(100.0, rel=0.05)


def test_no_dn_activity_means_baseline_walking():
    dec = EscapeDecoder(NAMES)
    assert np.allclose(dec.command(), [dec.base, dec.base])


def test_backward_dns_reverse_walking():
    dec = EscapeDecoder(NAMES)
    # DNp02 + DNp04 summing to r50 per side -> x_back = 1 -> halfway between walk and back up
    half = dec.r50 / 2
    s = dec.command(np.array([0, 0, half, half, half, half, 0, 0], float))
    assert s[0] == pytest.approx((dec.base - dec.v_back) / 2)
    s = dec.command(np.array([0, 0, 1e4, 1e4, 1e4, 1e4, 0, 0], float))
    assert s[0] == pytest.approx(-dec.v_back, abs=1e-2)


def test_forward_dn_speeds_up_and_giant_fiber_only_triggers_takeoff():
    dec = EscapeDecoder(NAMES)
    s = dec.command(np.array([0, 0, 0, 0, 0, 0, 1e4, 1e4], float))
    assert s[0] == pytest.approx(dec.v_fwd, abs=1e-2)
    gf = np.array([200, 200, 0, 0, 0, 0, 0, 0], float)
    assert dec.program_drive(gf)["takeoff"] is True
    assert np.allclose(dec.command(gf), dec.base)


def test_command_stays_in_the_controller_range():
    dec = EscapeDecoder(NAMES)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        s = dec.command(rng.exponential(100.0, len(NAMES)))[0]
        assert -dec.v_back - 1e-12 <= s <= max(dec.base, dec.v_fwd) + 1e-12


def test_decoder_step_end_to_end():
    dec = EscapeDecoder(NAMES)
    for _ in range(200):                                  # 200 ms of strong backward-DN firing
        s = dec.step(counts(DNp02=150, DNp04=150), 1.0)
    assert s[0] < -0.5


# ---- loop -------------------------------------------------------------------------------------
class FakeArena:
    """Stands in for FlyArena: a clock, a fixed retina, and a record of the commands."""
    dt_ms = 0.1

    def __init__(self):
        self.k, self.commands = 0, []

    @property
    def t_ms(self):
        return self.k * self.dt_ms

    def vision(self):
        return eyes(0, 0)

    def step(self, command, n):
        self.commands.append(np.asarray(command).copy())
        self.k += n

    def thorax(self):
        return np.zeros(3)

    head = ball = thorax


class FakeBrain:
    """Every watched DN fires on every tick from tick 3 on; records the rates it was given."""
    class params:
        dt = 0.1

    def __init__(self, n_watch):
        self.n_watch, self.calls, self.rates = n_watch, 0, []

    def step(self, n, rate_hz):
        self.rates.append(np.asarray(rate_hz).copy())
        self.calls += 1
        return np.full((self.n_watch, 1), 1 if self.calls > 3 else 0)


def test_loop_is_causal_and_runs_in_lockstep():
    """The command the body uses in tick k is the decoder's output from tick k-1, and brain and
    body advance by the same number of 0.1 ms steps each tick."""
    arena, brain = FakeArena(), FakeBrain(len(NAMES))
    dec = EscapeDecoder(NAMES)
    groups = [("LPLC2", "left", 2), ("LPLC2", "right", 2), ("LC4", "left", 1), ("LC4", "right", 1)]
    loop = ClosedLoop(arena, brain, LoomingEncoder(), dec, groups, tick_ms=1.0)
    log = loop.run(10.0).arrays()
    assert arena.k == 100 and brain.calls == 10                    # 10 ticks x 10 steps
    # The DNs first fire during tick 3 (0-based). Tick 3's body step must still use the old
    # command; the spikes may only change what the body does from tick 4 on.
    assert log["dn_counts"][:3].sum() == 0 and log["dn_counts"][3].sum() > 0
    assert np.allclose(log["command"][:4], dec.base)
    assert log["command"][4, 0] < dec.base
    assert np.allclose(np.array(arena.commands), log["command"])   # what the body really got
    assert brain.rates[0].shape == (6, 1)                          # one rate per driven neuron


def test_replay_holds_each_value_until_the_next():
    log = {"t": np.array([0.0, 1.0, 2.0]), "lplc2": np.array([[0, 0], [5, 6], [7, 8]]),
           "lc4": np.array([[1, 1], [2, 2], [3, 3]])}
    f = replay_from_log(log)
    assert f(0.5)["LPLC2"].tolist() == [0, 0]
    assert f(1.0)["LPLC2"].tolist() == [5, 6]
    assert f(1.99)["LC4"].tolist() == [2, 2]
    assert f(50.0)["LC4"].tolist() == [3, 3]


# ---- calibration on the real eyes (needs flygym) ------------------------------------------------
def test_size_estimate_matches_geometry_on_the_real_eyes():
    """A 2 mm black ball straight ahead of a standing fly at known distances: the encoder's size
    estimate must be within 20% of the true angular diameter 2 asin(r/d)."""
    pytest.importorskip("flygym")
    from c2c.control.body import FlyArena

    arena = FlyArena(ball_radius=2.0)
    enc = LoomingEncoder()
    head = arena.head()
    try:
        for d in [15.0, 8.0, 5.0]:
            arena.sim.mj_data.mocap_pos[0] = head + d * np.array([np.cos(0.35), 0, np.sin(0.35)])
            arena.sim.forward()
            est = enc.angular_size(enc.dark_area(arena.vision()).max())
            true = 2 * np.degrees(np.arcsin(2.0 / d))
            assert abs(est - true) / true < 0.2, (d, est, true)
    finally:
        arena.close()
