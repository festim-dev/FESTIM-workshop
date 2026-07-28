---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

# Openings

So far the only way gas entered or left an enclosure was through the contact surfaces. **Openings** let an
enclosure exchange gas with the outside world too, adding a term $Q$ to the pressure balance

$$
\frac{dP}{dt} = \frac{k_B T}{V} \left( \sum_\Gamma A_\Gamma \int_\Gamma \varphi \, dS
                                        + \sum_\text{openings} Q \right).
$$

FESTIM ships four openings:

| Opening | Flow rate $Q$ (particles/s) | Behaviour of a closed enclosure |
|---|---|---|
| `festim.Pump` | $-S\,P/(k_B T)$ | $P(t) = P_0\,e^{-S t / V}$ |
| `festim.Reservoir` | $C\,(P_\text{ext} - P)/(k_B T)$ | $P(t) = P_\text{ext} + (P_0 - P_\text{ext})\,e^{-C t / V}$ |
| `festim.PrescribedFlowRate` | $Q$ (imposed) | $P(t) = P_0 + Q\,k_B T\,t / V$ |
| `festim.EnclosureConnection` | $C\,(P_\text{other} - P)/(k_B T)$ | two enclosures equilibrate |

Each has a clean closed-form solution, which we use to verify the implementation below. All the
parameters (`pumping_speed`, `conductance`, `pressure`, `flow_rate`) can also be callables of time.

+++

## A minimal enclosure model

Openings act on the enclosure alone, so we don't need any transport physics: a single-material mesh with a
decoupled species (no boundary condition) is enough to carry the enclosure.

```{code-cell} ipython3
import numpy as np
import festim as F


def run_enclosure(openings, P0, species_name="H2", final_time=20.0, dt=1.0,
                  volume=1e-3, T=500.0):
    """Run a closed enclosure with the given openings and return its GasPressure export."""
    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = F.Mesh1D(np.linspace(0.0, 1.0, num=17))
    material = F.Material(name="mat", D_0=1e-9, E_D=0.0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=1.0)
    model.subdomains = [vol, left, right]

    H = F.Species("H", subdomains=[vol])
    model.species = [H]
    model.temperature = T

    H2 = F.GasSpecies(name=species_name, initial_pressure=P0)
    model.enclosures = [
        F.Enclosure(volume=volume, species=[H2], temperature=T, openings=openings)
    ]
    pressure = F.GasPressure(field=H2)
    model.exports = [pressure]
    model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=True,
                                final_time=final_time, stepsize=F.Stepsize(dt))
    model.show_progress_bar = False
    model.initialise()
    model.run()
    return H2, pressure


def backward_euler_decay(P0, rate, dt, nsteps):
    """Exact solution of the backward-Euler discretisation of dP/dt = -rate*P, so the
    check isolates the opening from the time-discretisation error."""
    return P0 * (1 + rate * dt) ** (-nsteps)
```

## Pump

A `festim.Pump` removes gas at a rate proportional to the pressure, $Q = -S\,P/(k_B T)$, so the
pressure of a closed enclosure decays exponentially.

```{code-cell} ipython3
P0, S, V = 1e5, 1e-4, 1e-3
dt, final_time = 1.0, 20.0

H2, pressure = run_enclosure([F.Pump(pumping_speed=S)], P0=P0, volume=V,
                             final_time=final_time, dt=dt)

nsteps = round(final_time / dt)
expected = backward_euler_decay(P0, S / V, dt, nsteps)
print(f"final pressure : {H2.value:.6e} Pa   (expected {expected:.6e})")
assert abs(H2.value - expected) / expected < 1e-8
```

## Reservoir

A `festim.Reservoir` connects the enclosure to an external volume held at a fixed pressure
`P_ext` through a conductance `C`. The pressure relaxes towards `P_ext`.

```{code-cell} ipython3
P0, P_ext, C, V = 1e5, 1e3, 1e-4, 1e-3

H2, pressure = run_enclosure([F.Reservoir(conductance=C, pressure=P_ext)], P0=P0, volume=V,
                             final_time=final_time, dt=dt)

expected = P_ext + backward_euler_decay(P0 - P_ext, C / V, dt, nsteps)
print(f"final pressure : {H2.value:.6e} Pa   (expected {expected:.6e})")
assert abs(H2.value - expected) / expected < 1e-8
```

```{admonition} Prefer pressure-proportional openings for pumping
:class: warning
Both `Pump` and `Reservoir` are proportional to the pressure, so they self-regulate and cannot drive the
pressure negative. A `festim.PrescribedFlowRate` with a *negative* (removal) rate is
pressure-independent: it keeps pulling gas out even after the enclosure has emptied, which produces
unphysical negative pressures. Use `PrescribedFlowRate` for a controlled injection, and `Pump`/`Reservoir`
for pumping.
```

## PrescribedFlowRate

A `festim.PrescribedFlowRate` imposes the flow rate directly, independent of the pressure. A
constant injection fills the enclosure linearly.

```{code-cell} ipython3
P0, Q, V, T = 1e3, 1e18, 1e-3, 500.0

H2, pressure = run_enclosure([F.PrescribedFlowRate(flow_rate=Q)], P0=P0, volume=V,
                             final_time=final_time, dt=dt, T=T)

expected = P0 + Q * F.k_B_SI * T * final_time / V
print(f"final pressure : {H2.value:.6e} Pa   (expected {expected:.6e})")
assert abs(H2.value - expected) / expected < 1e-8
```

## EnclosureConnection

An `festim.EnclosureConnection` couples two enclosures through a conductance. Their pressure
difference decays as $\exp\!\big(-C(1/V_1 + 1/V_2)\,t\big)$ while the volume-weighted mean is conserved.
The connection is declared in the openings of **one** enclosure only, and the mirror term is added to the
partner automatically.

```{code-cell} ipython3
P1_0, P2_0 = 1e5, 0.0
V1, V2, C = 1e-3, 2e-3, 1e-4

H2_a = F.GasSpecies(name="H2_a", initial_pressure=P1_0)
H2_b = F.GasSpecies(name="H2_b", initial_pressure=P2_0)
connection = F.EnclosureConnection(conductance=C, species=(H2_a, H2_b))

enclosure_a = F.Enclosure(volume=V1, species=[H2_a], temperature=500.0, openings=[connection])
enclosure_b = F.Enclosure(volume=V2, species=[H2_b], temperature=500.0)

model = F.HydrogenTransportProblemDiscontinuous()
model.mesh = F.Mesh1D(np.linspace(0.0, 1.0, num=17))
material = F.Material(name="mat", D_0=1e-9, E_D=0.0)
vol = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
model.subdomains = [vol, F.SurfaceSubdomain1D(id=1, x=0.0), F.SurfaceSubdomain1D(id=2, x=1.0)]
model.species = [F.Species("H", subdomains=[vol])]
model.temperature = 500.0
model.enclosures = [enclosure_a, enclosure_b]
p_a = F.GasPressure(field=H2_a)
p_b = F.GasPressure(field=H2_b)
model.exports = [p_a, p_b]
model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=True,
                            final_time=final_time, stepsize=F.Stepsize(dt))
model.show_progress_bar = False
model.initialise()
model.run()

P1, P2 = H2_a.value, H2_b.value
expected_diff = backward_euler_decay(P1_0 - P2_0, C * (1 / V1 + 1 / V2), dt, nsteps)
print(f"P1 = {P1:.4e} Pa,  P2 = {P2:.4e} Pa")
assert abs((P1 - P2) - expected_diff) / expected_diff < 1e-8            # difference decays
assert abs((P1 * V1 + P2 * V2) - (P1_0 * V1 + P2_0 * V2)) / (P1_0 * V1) < 1e-8  # mean conserved
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

plt.plot(p_a.t, p_a.data, label="enclosure a")
plt.plot(p_b.t, p_b.data, label="enclosure b")
plt.xlabel("time (s)")
plt.ylabel("pressure (Pa)")
plt.title("Two enclosures equilibrating through a connection")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```
