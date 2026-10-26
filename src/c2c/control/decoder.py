"""Motor decoding: descending-neuron spikes -> a walking command for the simulated body.

The brain's output to the body is about 1,300 descending neurons (DNs). We read eight of them, the
four escape types on each side, and turn their spikes into the two numbers flygym's
``HybridTurningController`` accepts: one "descending signal" per side of the body. Its magnitude
sets stepping amplitude (|s| = 1 is normal walking) and its sign sets direction (negative = the
CPG runs backwards, i.e. backward walking).

Step 1: spikes -> rates
-----------------------
A single spike says little; a rate does. Each DN's rate is estimated with an exponential filter
(time constant tau_ms = 20 ms) on its spike count in every control tick:

    r <- r + (1 - exp(-dt/tau)) * (count/dt - r)

which is the same leaky integration a downstream neuron or a muscle performs. tau sets the
trade-off: shorter = faster but noisier command, longer = smoother but later.

Step 2: rates -> motor program weights (the assumptions, each one stated)
-------------------------------------------------------------------------
A1. DNp02 + DNp04 drive BACKWARD escape. Co-activating them makes flies jump backwards, and
    DNp02 shifts the body backwards before takeoff (Peek 2018, PhD thesis, U. Chicago;
    Dombrovski et al. 2023, Nature).
A2. DNp11 drives FORWARD escape: activating it gives forward jumps (same sources).
A3. DNp01, the giant fiber, triggers the short-mode TAKEOFF (wing depression + leg extension).
    The flygym body here is legs-only with a walking controller, so it cannot take off. The giant
    fiber's output is logged as a takeoff trigger, not executed.
A4. Escape on foot. Takeoff direction is expressed as walking direction: backward escape ->
    backward walking, forward escape -> forward walking at escape speed.
A5. Baseline walking is a constant forward drive. The DNs that drive normal walking are not
    stimulated in these experiments, so a tonic command stands in for "the fly was walking".
A6. Competition. Walking and the two escape programs are mixed as a weighted average in which
    walking has weight 1 and each escape program has weight x = (its DN rate) / r50:

        s = (base + x_fwd * v_fwd - x_back * v_back) / (1 + x_fwd + x_back)

    With no DN activity s = base (walk). When backward-escape DNs fire at r50 Hz, the fly is
    halfway between walking and backing up. Far above r50, backward escape takes over completely.
    No gains to tune except r50, and s stays inside [-v_back, max(base, v_fwd)], which keeps the
    walking controller in its stable range.
A7. Both sides get the same command (no turning): the stimulus in our experiments is frontal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BACKWARD = ("DNp02", "DNp04")
FORWARD = ("DNp11",)
TAKEOFF = ("DNp01",)


@dataclass
class EscapeDecoder:
    neurons: list[str]             # names in the order of the spike counts, e.g. "DNp04 L"
    tau_ms: float = 20.0           # rate filter time constant
    r50: float = 40.0              # Hz, rate at which an escape program gets weight 1
    base: float = 1.0              # tonic walking command (A5)
    v_back: float = 1.2            # backward-escape command magnitude
    v_fwd: float = 1.2             # forward-escape command magnitude
    takeoff_hz: float = 50.0       # giant-fiber rate counted as a takeoff trigger (A3)

    rates: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        types = [n.split(" ")[0] for n in self.neurons]
        self._back = np.isin(types, BACKWARD)
        self._fwd = np.isin(types, FORWARD)
        self._gf = np.isin(types, TAKEOFF)
        self.reset()

    def reset(self) -> None:
        self.rates = np.zeros(len(self.neurons))

    def update_rates(self, counts: np.ndarray, dt_ms: float) -> np.ndarray:
        """Spike counts of each neuron in the last tick (length = len(neurons)) -> rates (Hz)."""
        counts = np.asarray(counts, dtype=float).reshape(len(self.neurons))
        alpha = 1.0 - np.exp(-dt_ms / self.tau_ms)
        self.rates = self.rates + alpha * (counts / (dt_ms / 1000.0) - self.rates)
        return self.rates

    def program_drive(self, rates: np.ndarray | None = None) -> dict[str, float]:
        """Rates -> x_back, x_fwd (escape program weights, A1, A2) and the takeoff flag (A3).
        Per program, DN rates are summed within a side (DNp02 + DNp04) and averaged over the
        two sides."""
        r = self.rates if rates is None else np.asarray(rates, dtype=float)
        x_back = r[self._back].sum() / 2.0 / self.r50
        x_fwd = r[self._fwd].sum() / 2.0 / self.r50
        takeoff = bool(np.any(r[self._gf] > self.takeoff_hz))
        return {"x_back": float(x_back), "x_fwd": float(x_fwd), "takeoff": takeoff}

    def command(self, rates: np.ndarray | None = None) -> np.ndarray:
        """Descending signal (left, right) for HybridTurningController (A4-A7)."""
        p = self.program_drive(rates)
        s = (self.base + p["x_fwd"] * self.v_fwd - p["x_back"] * self.v_back) / (
            1.0 + p["x_fwd"] + p["x_back"])
        return np.array([s, s])

    def step(self, counts: np.ndarray, dt_ms: float) -> np.ndarray:
        self.update_rates(counts, dt_ms)
        return self.command()
