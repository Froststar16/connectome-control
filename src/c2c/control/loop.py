"""The sensorimotor loop: body sensors -> sensory neurons -> brain -> descending neurons -> body.

Two simulators run side by side, each with its own state, and exchange messages once per
control tick (default 1 ms = 10 brain steps = 10 physics steps):

    tick k (time t = k * tick):
      1. SENSE   every vision period, render the eyes and update the encoder -> LPLC2/LC4 rates
      2. THINK   brain.step(10, rates): the spiking brain advances t -> t + tick with those rates
                 held constant, and reports how often each escape DN fired
      3. DECODE  decoder turns the DN counts into the next walking command
      4. ACT     the body advances t -> t + tick with the command decided at the END OF THE
                 PREVIOUS TICK, then the new command takes over

Why one tick of delay in step 4? Causality. The DN spikes of tick k happen somewhere between t and
t + tick; letting them steer the body during that same interval would act on them before they
exist. Holding the old command for one tick makes the loop strictly causal, at the cost of
<= 1 ms extra latency, small next to the brain's own ~5 ms from looming input to the first DN spike.

Latency budget from "the ball got bigger" to "the legs move differently" (defaults):
    vision sampling        <= 2 ms   (one vision frame; flies see ~ 100-200 Hz flicker)
    encoder filter         ~ 15 ms   (size low-pass; the velocity estimate lags by about tau)
    brain                  ~ 5-10 ms (LPLC2/LC4 -> DNp04 first, then DNp02/DNp11; Phase 3)
    decoder rate filter    ~ 20 ms   (tau of the DN rate estimate)
    control tick           <= 1 ms
    CPG amplitude          ~ 50 ms   (oscillator magnitudes converge at 20 /s)
The filters dominate, not the neurons: a design lesson that also holds for robots.

Modes
-----
closed     the encoder sees the live eyes, so the body's own movement changes its input
open       a recorded rate trace is replayed into the brain; the body moves but cannot change
           what the brain receives
no_brain   the brain is skipped and the decoder sees no DN activity (baseline behaviour)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from c2c.control.encoder import rates_for_drive


@dataclass
class LoopLog:
    """One row per control tick. Saved and loaded as a flat .npz."""
    t: list = field(default_factory=list)              # ms, start of the tick
    lplc2: list = field(default_factory=list)          # Hz (left, right) applied this tick
    lc4: list = field(default_factory=list)
    theta: list = field(default_factory=list)          # deg, encoder size estimate
    velocity: list = field(default_factory=list)       # deg/s
    dn_counts: list = field(default_factory=list)      # spikes per watched DN this tick
    dn_rates: list = field(default_factory=list)       # Hz, decoder's filtered rates
    command: list = field(default_factory=list)        # descending signal applied this tick
    takeoff: list = field(default_factory=list)        # giant-fiber takeoff trigger
    thorax: list = field(default_factory=list)         # mm, end of tick
    head: list = field(default_factory=list)
    ball: list = field(default_factory=list)
    yaw: list = field(default_factory=list)            # deg, thorax heading at end of tick
    drive: list = field(default_factory=list)          # Hz per driven neuron (if log_drive)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


class ClosedLoop:
    def __init__(self, arena, brain, encoder, decoder, drive_groups, mode: str = "closed",
                 replay=None, tick_ms: float = 1.0, vision_ms: float = 2.0,
                 log_drive: bool = False):
        """
        arena        c2c.control.body.FlyArena
        brain        c2c.sim.torch_lif.LIFStepper, watching the decoder's DNs in order
                     (n_trials=1), or None for mode="no_brain"
        encoder      c2c.control.encoder.LoomingEncoder
        decoder      c2c.control.decoder.EscapeDecoder
        drive_groups from encoder.looming_drive, matching the brain's drive indices
        replay       for mode="open": function t_ms -> {"LPLC2": (2,), "LC4": (2,)}
        log_drive    also log the rate of every driven neuron (Phase 5, retinotopic input)

        The encoder turns its own output into one rate per driven neuron
        (``encoder.neuron_rates``): the same rate for a whole side (LoomingEncoder) or one per
        neuron (RetinotopicEncoder). A replayed trace is always per side.
        """
        if mode not in ("closed", "open", "no_brain"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "open" and replay is None:
            raise ValueError("open loop needs a replay function")
        self.arena, self.brain, self.encoder, self.decoder = arena, brain, encoder, decoder
        self.groups, self.mode, self.replay = drive_groups, mode, replay
        self.n_tick = round(tick_ms / arena.dt_ms)          # physics steps per tick
        if brain is not None and round(tick_ms / brain.params.dt) != self.n_tick:
            raise ValueError("brain dt and physics timestep must be equal")
        self.n_vision = round(vision_ms / tick_ms)          # ticks per vision frame
        self.tick_ms = tick_ms
        self.log_drive = log_drive

    def run(self, t_run_ms: float, verbose: bool = False) -> LoopLog:
        log = LoopLog()
        zero = {"LPLC2": np.zeros(2), "LC4": np.zeros(2)}
        rates = zero
        command = np.array([self.decoder.base] * 2)           # walk until the brain says otherwise
        n_dn = len(self.decoder.neurons)
        wall = {"vision": 0.0, "brain": 0.0, "body": 0.0}
        n_ticks = round(t_run_ms / self.tick_ms)
        for k in range(n_ticks):
            t = self.arena.t_ms
            # 1. SENSE
            c0 = time.perf_counter()
            if self.mode == "open":
                rates = self.replay(t)
            elif k % self.n_vision == 0:          # closed loop and no_brain: live eyes
                rates = self.encoder.update(self.arena.vision(), t)
            c1 = time.perf_counter()
            # 2. THINK
            drive_hz = None
            if self.groups:                       # no groups: nothing is driven (no_brain)
                drive_hz = (rates_for_drive(self.groups, rates) if self.mode == "open"
                            else self.encoder.neuron_rates(rates, self.groups))
            if self.brain is not None:
                counts = self.brain.step(self.n_tick, rate_hz=drive_hz[:, None])[:, 0]
            else:
                counts = np.zeros(n_dn, dtype=np.int64)
            c2 = time.perf_counter()
            # 3. DECODE (takes effect next tick)
            next_command = self.decoder.step(counts, self.tick_ms)
            drive = self.decoder.program_drive()
            # 4. ACT with the previous tick's command
            self.arena.step(command, self.n_tick)
            c3 = time.perf_counter()
            wall["vision"] += c1 - c0
            wall["brain"] += c2 - c1
            wall["body"] += c3 - c2

            log.t.append(t)
            log.lplc2.append(np.asarray(rates["LPLC2"], float).copy())
            log.lc4.append(np.asarray(rates["LC4"], float).copy())
            log.theta.append(self.encoder.theta)
            log.velocity.append(self.encoder.velocity)
            log.dn_counts.append(counts.copy())
            log.dn_rates.append(self.decoder.rates.copy())
            log.command.append(command.copy())
            log.takeoff.append(drive["takeoff"])
            log.thorax.append(self.arena.thorax())
            log.head.append(self.arena.head())
            log.ball.append(self.arena.ball())
            if hasattr(self.arena, "heading"):
                log.yaw.append(self.arena.heading())
            if self.log_drive and drive_hz is not None:
                log.drive.append(np.asarray(drive_hz, dtype=np.float32).copy())
            command = next_command
            if verbose and (k + 1) % max(1, n_ticks // 10) == 0:
                print(f"  {t + self.tick_ms:6.0f} ms  wall " +
                      "  ".join(f"{a} {b:.1f} s" for a, b in wall.items()))
        self.wall = wall
        return log


def replay_from_log(log: dict[str, np.ndarray]):
    """A recorded encoder output (LoopLog arrays) as a function t_ms -> rates (zero-order hold)."""
    t = np.asarray(log["t"])
    lplc2, lc4 = np.asarray(log["lplc2"]), np.asarray(log["lc4"])

    def rates(t_ms: float) -> dict[str, np.ndarray]:
        i = int(np.searchsorted(t, t_ms + 1e-9, side="right")) - 1
        if i < 0:
            return {"LPLC2": np.zeros(2), "LC4": np.zeros(2)}
        return {"LPLC2": lplc2[i], "LC4": lc4[i]}

    return rates

