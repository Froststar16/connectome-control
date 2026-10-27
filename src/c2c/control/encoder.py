"""Sensory encoding: what the fly's eyes see -> Poisson rates for the looming detectors.

The simulated compound eye (flygym) gives one brightness value per ommatidium, 721 per eye.
The brain model wants something else: a firing rate for each of the 314 LPLC2 and LC4 neurons it
drives with Poisson input. This module is the translation, kept deliberately simple so that every
number can be explained.

What LPLC2 and LC4 respond to (electrophysiology)
-------------------------------------------------
Both are visual projection neurons that respond to looming, an object growing on the retina as it
approaches. They carry different features of it:

* LC4 encodes the angular VELOCITY of expansion, how fast the object's edges move apart
  (von Reyn et al. 2017, Neuron).
* LPLC2 encodes the angular SIZE of a looming object (Ache et al. 2019, Curr. Biol.) and only
  responds to expansion: a large object that is not growing does not drive it.

The giant fiber sums the two. Here we compute both features and turn each into a rate.

The detector, step by step
--------------------------
1. Darkness. The looming object is black; everything else in the arena (white sky, grey checker
   floor) is brighter than ~0.14. Each ommatidium gets a soft "covered by a dark object" score

       c = clip((hi - I) / (hi - lo), 0, 1)        I = brightness, lo = 0.02, hi = 0.08

   which is 1 for pure black, 0 for anything as bright as the floor, in between at edges.
2. Size. A = sum of c over one eye = dark area in ommatidia. One ommatidium sees about a
   DPHI x DPHI patch of the world, so a dark disk of angular diameter theta covers about
   (theta / DPHI)^2 ommatidia: theta ~= DPHI * sqrt(A). DPHI = 5.5 deg was calibrated by placing a
   2 mm ball at known distances in front of a standing fly (real Drosophila ommatidia are about
   5 deg apart). The object's size is the larger of the two eyes' estimates: a frontal object
   sits on the border between the eyes, and as the head sways during walking it hops from one eye
   to the other, so either eye alone sees it flicker in and out.
3. Expansion speed. theta is low-pass filtered (tau_ms) because it moves in ommatidium-sized
   jumps, then differentiated between vision frames: v = d(theta)/dt in deg/s.
4. Rates. Each feature is mapped linearly onto 0 ... r_max between a threshold and a saturation
   value. Nothing responds until the object covers more than ~1.5 ommatidia (theta_min), so that
   a far-away speck flickering across one ommatidium is not mistaken for looming:

       r_LC4   = r_max * clip(v / v_sat, 0, 1)                               * [theta > theta_min]
       r_LPLC2 = r_max * clip((theta - theta_0) / (theta_sat - theta_0), 0, 1) * [v > v_gate]

5. Sides. Each eye's neurons get these rates times the eye's share of the object,
   A_eye / max(A_left, A_right): a frontal object drives both sides, a lateral one mostly one.
   Within one side, every LPLC2 (or LC4) neuron gets the same rate. That ignores the retinotopy
   of these cells (each one sees its own patch of the eye), the main simplification here.
   Phase 5 removes it: see ``RetinotopicEncoder`` at the end of this file.

Left eye -> left-side neurons: LPLC2 and LC4 are optic-lobe cells, and the annotations' "side" is
the side of the brain they sit in, which is the side of the eye they receive from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EYES = ("left", "right")


@dataclass
class LoomingEncoder:
    """Ommatidia readouts -> object size and expansion speed -> LPLC2 / LC4 rates per eye."""

    r_max: float = 150.0          # Hz, Poisson rate at saturation (Phase 3 swept 10-200 Hz)
    dark_lo: float = 0.02         # brightness counted as fully dark
    dark_hi: float = 0.08         # brightness counted as not dark at all
    dphi_deg: float = 5.5         # degrees of visual angle per ommatidium (calibrated)
    tau_ms: float = 15.0          # low-pass time constant of the size estimate
    theta_min: float = 8.0        # deg, smallest object that drives anything
    theta_0: float = 10.0         # deg, LPLC2 threshold size
    theta_sat: float = 60.0       # deg, LPLC2 saturation size
    v_sat: float = 500.0          # deg/s, LC4 saturation speed
    v_gate: float = 20.0          # deg/s, minimum expansion speed for LPLC2 to respond

    # state
    theta: float = 0.0                                          # filtered size, deg
    velocity: float = 0.0                                       # deg/s
    share: np.ndarray = field(default_factory=lambda: np.zeros(2))   # per-eye share of object
    _last_t: float | None = None

    def reset(self) -> None:
        self.theta, self.velocity, self._last_t = 0.0, 0.0, None
        self.share = np.zeros(2)

    # -- the stages, separately testable --------------------------------------------------------
    def dark_area(self, ommatidia: np.ndarray) -> np.ndarray:
        """(2, n_ommatidia, 2) flygym readout -> dark area of each eye in ommatidia, shape (2,).
        Each ommatidium is either yellow- or pale-type and the other channel is 0, so the max over
        the last axis is its brightness."""
        brightness = np.asarray(ommatidia).max(axis=-1)
        covered = np.clip((self.dark_hi - brightness) / (self.dark_hi - self.dark_lo), 0.0, 1.0)
        return covered.sum(axis=-1)

    def angular_size(self, area) -> np.ndarray:
        """Dark area (ommatidia) -> estimated angular diameter in degrees."""
        return self.dphi_deg * np.sqrt(np.asarray(area, dtype=float))

    def rates(self, theta: float, velocity: float, share=(1.0, 1.0)) -> dict[str, np.ndarray]:
        """Features -> {"LPLC2": (2,), "LC4": (2,)} rates in Hz, (left eye, right eye)."""
        share = np.asarray(share, dtype=float)
        lc4 = self.r_max * np.clip(velocity / self.v_sat, 0.0, 1.0) * (theta > self.theta_min)
        size = np.clip((theta - self.theta_0) / (self.theta_sat - self.theta_0), 0.0, 1.0)
        lplc2 = self.r_max * size * (velocity > self.v_gate)
        return {"LPLC2": lplc2 * share, "LC4": lc4 * share}

    # -- the whole thing, called once per vision frame ------------------------------------------
    def update(self, ommatidia: np.ndarray, t_ms: float) -> dict[str, np.ndarray]:
        """Feed one vision frame taken at time t_ms; returns the rates to apply until the next."""
        area = self.dark_area(ommatidia)
        size = float(self.angular_size(area.max()))
        self.share = area / area.max() if area.max() > 0 else np.zeros(2)
        if self._last_t is None:
            self.theta = size
        elif t_ms > self._last_t:
            dt = t_ms - self._last_t
            alpha = 1.0 - np.exp(-dt / self.tau_ms)              # exact first-order filter
            new_theta = self.theta + alpha * (size - self.theta)
            self.velocity = (new_theta - self.theta) / (dt / 1000.0)
            self.theta = new_theta
        self._last_t = t_ms
        return self.rates(self.theta, self.velocity, self.share)

    def neuron_rates(self, rates: dict, groups) -> np.ndarray:
        """Encoder output -> one rate per driven neuron (the loop calls this). Uniform within a
        side: every neuron of a (type, side) group gets the group's rate."""
        return rates_for_drive(groups, rates)


def looming_drive(aligned, types=("LPLC2", "LC4")):
    """Neuron indices to drive, grouped by (cell type, side).

    Returns (drive, groups): ``drive`` is the concatenation of the groups' neuron indices, and
    groups[k] = (type, side, n_neurons) in the same order. ``rates_for_drive`` uses the groups to
    turn encoder output into one rate per driven neuron.
    """
    from c2c.data.annotations import select

    parts, groups = [], []
    for t in types:
        for side in EYES:
            idx = np.asarray(select(aligned, cell_type=t, side=side), dtype=np.int64)
            parts.append(idx)
            groups.append((t, side, len(idx)))
    return np.concatenate(parts), groups


def rates_for_drive(groups, rates: dict[str, np.ndarray]) -> np.ndarray:
    """Encoder output {"LPLC2": (2,), "LC4": (2,)} -> one rate per driven neuron, in the order of
    ``looming_drive``."""
    return np.concatenate([np.full(n, float(rates[t][EYES.index(side)]))
                           for t, side, n in groups])


# ---- Phase 5: every neuron its own receptive field ----------------------------------------------
@dataclass
class RetinotopicEncoder(LoomingEncoder):
    """Same looming features as ``LoomingEncoder``, but each neuron sees only its own patch of eye.

    Receptive fields
    ----------------
    Neuron i has a receptive field (RF) centred on direction u_i with width sigma_i, both measured
    from the connectome (c2c.data.eyemap). Ommatidium j looks along d_j (flygym's eye geometry,
    ``FlyArena.ommatidia_directions``). The weight of ommatidium j in neuron i's RF is a Gaussian
    of the angle between them:

        w_ij = exp( -angle(u_i, d_j)^2 / (2 sigma_i^2) )         (own eye only)

    The rate: separable "what x where"
    ----------------------------------
    The rate of neuron i is the same feature tuning as in Phase 4 (WHAT: size for LPLC2,
    expansion speed for LC4, computed from the whole object) times a spatial factor (WHERE):

        rate_i = f_type(theta, v) * h_i,       h_i = max_j ( w_ij * c_j )

    c_j is ommatidium j's "covered by a dark object" score. h_i asks: how close to my RF centre
    does the object reach? It is 1 when the object covers the centre, exp(-a^2 / 2 sigma^2) for
    a small object a degrees away (the classic Gaussian tuning curve), and 0 when nothing dark is
    in view. Taking the max rather than a sum keeps h_i between 0 and 1, so the strongest rate
    any neuron can get is the Phase 4 rate and the two encoders are directly comparable.

    Assumption R1: rate = feature tuning x spatial tuning (a separable RF). Real LPLC2 needs
    outward motion on all four sides of its RF centre (Klapoetke et al. 2017); the max-overlap
    rule approximates "the looming object is centred on me" without simulating motion detectors.
    Assumption R2: the RF frame of the eye map is the head frame of the body model.
    Assumption R3: the connectome gives each RF's CENTRE and relative SIZE, physiology its
    absolute width. sigma from the connectome is the spread of a neuron's direct columnar
    inputs, but those inputs (T4, T5, Tm cells) each pool several columns themselves, so the
    functional RF is wider. ``sigma_scale`` multiplies every sigma; Phase 5 sets it so that the
    median LPLC2 RF is 60 deg across at half maximum, the size measured by Klapoetke et al. 2017.

    Population coding
    -----------------
    No single neuron says where the object is: one neuron's rate drops both when the object is
    smaller and when it is further from the RF centre. The position is in the PATTERN across
    the population: which neurons are active. The brain can read it out because each LC4 or
    LPLC2 neuron connects to the descending neurons with its own weight (Phase 5 checks the
    Dombrovski et al. 2023 gradients in the connectome).
    """

    rf_dirs: np.ndarray | None = None      # (n, 3) RF centres, head frame, in drive order
    rf_sigma: np.ndarray | None = None     # (n,) degrees
    eye: np.ndarray | None = None          # (n,) 0 = left eye, 1 = right eye
    cell_type: np.ndarray | None = None    # (n,) "LPLC2" / "LC4"
    omm_dirs: np.ndarray | None = None     # (2, n_ommatidia, 3), head frame
    sigma_scale: float = 1.0               # functional / connectome RF width (R3)
    h: np.ndarray = field(default_factory=lambda: np.zeros(0))   # last spatial factors

    def __post_init__(self) -> None:
        for name in ("rf_dirs", "rf_sigma", "eye", "cell_type", "omm_dirs"):
            if getattr(self, name) is None:
                raise ValueError(f"RetinotopicEncoder needs {name}")
        self.rf_dirs = np.asarray(self.rf_dirs, dtype=float)
        self.rf_sigma = np.asarray(self.rf_sigma, dtype=float)
        self.eye = np.asarray(self.eye, dtype=np.int64)
        self.cell_type = np.asarray(self.cell_type)
        # w[i, j]: weight of ommatidium j (of neuron i's own eye) in neuron i's RF
        d = np.asarray(self.omm_dirs, dtype=float)[self.eye]              # (n, n_omm, 3)
        cos = np.clip(np.einsum("nk,njk->nj", self.rf_dirs, d), -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        sigma = self.sigma_scale * self.rf_sigma[:, None]
        self.weights = np.exp(-ang**2 / (2.0 * sigma**2))
        self.h = np.zeros(len(self.rf_sigma))

    def covered(self, ommatidia: np.ndarray) -> np.ndarray:
        """(2, n_ommatidia, 2) readout -> (2, n_ommatidia) dark-coverage score c in [0, 1]."""
        brightness = np.asarray(ommatidia).max(axis=-1)
        return np.clip((self.dark_hi - brightness) / (self.dark_hi - self.dark_lo), 0.0, 1.0)

    def spatial(self, ommatidia: np.ndarray) -> np.ndarray:
        """The WHERE factor h_i for every neuron: max_j w_ij c_j over its own eye."""
        c = self.covered(ommatidia)[self.eye]                              # (n, n_omm)
        return (self.weights * c).max(axis=1)

    def update(self, ommatidia: np.ndarray, t_ms: float) -> dict[str, np.ndarray]:
        """One vision frame -> {"LPLC2": (2,), "LC4": (2,), "neurons": (n,)}.

        "neurons" is what drives the brain. The per-side entries are the MEAN rate of each
        (type, side) group, kept so the log has the same shape as in Phase 4."""
        super().update(ommatidia, t_ms)                       # updates theta, velocity, share
        feature = self.rates(self.theta, self.velocity)       # per type, before any spatial factor
        self.h = self.spatial(ommatidia)
        per_type = np.where(self.cell_type == "LC4", feature["LC4"][0], feature["LPLC2"][0])
        neurons = per_type * self.h
        out = {"neurons": neurons}
        for t in ("LPLC2", "LC4"):
            m = self.cell_type == t
            out[t] = np.array([neurons[m & (self.eye == e)].mean() if np.any(m & (self.eye == e))
                               else 0.0 for e in (0, 1)])
        return out

    def neuron_rates(self, rates: dict, groups) -> np.ndarray:
        if "neurons" not in rates:                 # before the first vision frame
            return np.zeros(len(self.rf_sigma))
        return rates["neurons"]


def retinotopic_from_table(rf, drive_root_ids, omm_dirs, **kw) -> RetinotopicEncoder:
    """Build a RetinotopicEncoder whose neurons are in drive order.

    rf               receptive-field table (c2c.data.eyemap.receptive_fields): root_id,
                     cell_type, side, x, y, z, sigma
    drive_root_ids   FlyWire IDs of the driven neurons, in the order of ``looming_drive``
    omm_dirs         FlyArena.ommatidia_directions()
    """
    table = rf.set_index("root_id").loc[np.asarray(drive_root_ids, dtype=np.int64)]
    if table[["x", "y", "z", "sigma"]].isna().any().any():
        raise ValueError("some driven neurons have no receptive field")
    return RetinotopicEncoder(
        rf_dirs=table[["x", "y", "z"]].to_numpy(float), rf_sigma=table["sigma"].to_numpy(float),
        eye=np.where(table["side"].to_numpy() == "left", 0, 1),
        cell_type=table["cell_type"].to_numpy(), omm_dirs=omm_dirs, **kw)
