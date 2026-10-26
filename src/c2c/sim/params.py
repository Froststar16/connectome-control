"""Constants of the whole-brain leaky integrate-and-fire (LIF) model of Shiu et al. (Nature 2024).

Every value below is copied from ``default_params`` in the authors' code
(github.com/philshiu/Drosophila_brain_model, model.py), with the source each value came from:

    v_0, v_rst   -52 mV   resting and reset potential      Kakaria & de Bivort 2017
    v_th         -45 mV   spike threshold                  Kakaria & de Bivort 2017
    t_mbr         20 ms   membrane time constant (R*C)     Kakaria & de Bivort 2017
    tau            5 ms   synaptic time constant           Juergensen et al. 2021
    t_rfc        2.2 ms   refractory period                Lazar et al. 2021
    t_dly        1.8 ms   synaptic delay                   Paul et al. 2015
    w_syn      0.275 mV   weight of ONE synapse            free parameter (fitted, see below)
    f_poi        250      Poisson kick = 250 * w_syn       "sufficient to cause spiking"
    dt           0.1 ms   time step                        Brian2 default (the authors never set it)

The model, for every neuron i:

    dv/dt = (v_0 - v + g) / t_mbr          membrane potential relaxes to rest, pushed by g
    dg/dt = -g / tau                        synaptic drive decays exponentially
    spike when v > v_th  ->  v = v_rst, g = 0, then frozen for t_rfc
    each spike of presynaptic j adds  w_syn * W[i, j]  to g_i, t_dly later

``W[i, j]`` is a signed synapse COUNT, so a 20-synapse excitatory connection adds 20 * 0.275 =
5.5 mV to g. ``g`` has units of voltage: it is the voltage the membrane would be pushed to if the
input were held constant. That is why the model needs only ONE free parameter: w_syn converts
"number of synapses" into "millivolts", and the authors tuned it so that sugar neurons firing at
100 Hz drive the feeding motor neuron MN9 to roughly 80% of its maximum rate.

All quantities are plain floats in ms, mV and Hz so that both simulators (Brian2 with units,
PyTorch without) read them from one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LIFParams:
    v_0: float = -52.0     # mV
    v_rst: float = -52.0   # mV
    v_th: float = -45.0    # mV
    t_mbr: float = 20.0    # ms
    tau: float = 5.0       # ms
    t_rfc: float = 2.2     # ms
    t_dly: float = 1.8     # ms
    w_syn: float = 0.275   # mV per synapse
    f_poi: float = 250.0   # Poisson kick in units of w_syn
    dt: float = 0.1        # ms

    # -- derived quantities ---------------------------------------------------------------
    @property
    def w_poi(self) -> float:
        """Size of one Poisson kick to v (mV). 68.75 mV is ~10x the 7 mV gap between rest and
        threshold, so every Poisson event forces a spike: the stimulated neuron simply fires as
        a Poisson process at the chosen rate."""
        return self.w_syn * self.f_poi

    def steps(self, t_ms: float) -> int:
        """Duration in ms -> whole number of time steps. Rounded (not truncated) so that
        1.8 ms / 0.1 ms = 17.999999999999996 becomes 18, the same rule Brian2 uses."""
        return round(t_ms / self.dt)

    @property
    def delay_steps(self) -> int:
        return self.steps(self.t_dly)

    @property
    def refractory_steps(self) -> int:
        return self.steps(self.t_rfc)

    def propagators(self) -> tuple[float, float, float]:
        """Exact one-step solution of the linear ODE pair (no Euler error).

        With u = v - v_0 and no spikes during the step, the equations are linear with constant
        coefficients, so they can be solved in closed form over one step dt:

            g(t+dt) = P_gg * g(t)
            u(t+dt) = P_vv * u(t) + P_vg * g(t)

            P_gg = exp(-dt/tau)
            P_vv = exp(-dt/t_mbr)
            P_vg = tau / (tau - t_mbr) * (exp(-dt/tau) - exp(-dt/t_mbr))

        Derivation: g(t) = g0 exp(-t/tau) is a known forcing term, so du/dt = -u/t_mbr + g/t_mbr
        is a first-order linear ODE. Try a particular solution A exp(-t/tau), solve for A, add the
        homogeneous solution, match u(0) = u0. This is "exact integration" (Rotter & Diesmann
        1999), the same thing Brian2 does for ``method='linear'``. It is exact for ANY dt; dt only
        controls how finely spike times are resolved.
        """
        p_gg = math.exp(-self.dt / self.tau)
        p_vv = math.exp(-self.dt / self.t_mbr)
        if math.isclose(self.tau, self.t_mbr):
            p_vg = (self.dt / self.tau) * p_gg  # limit of the formula as tau -> t_mbr
        else:
            p_vg = self.tau / (self.tau - self.t_mbr) * (p_gg - p_vv)
        return p_vv, p_vg, p_gg

    def with_(self, **changes) -> LIFParams:
        """Copy with some fields changed, e.g. ``params.with_(w_syn=0.3)``."""
        return replace(self, **changes)


SHIU2024 = LIFParams()
