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

# Coupling modes

An enclosure can be coupled to the solid in two ways:

* **Flux coupling** — a `festim.SurfaceReactionBC` whose `gas_pressure` is the `GasSpecies`.
  The gas exchanges with the solid through an explicit surface reaction
  ($2\,\mathrm{H} \rightleftharpoons \mathrm{H_2}$), as in [](enclosure_intro.md). Use this when the
  recombination/dissociation kinetics matter.
* **Dirichlet (concentration) coupling** — a `festim.SievertsBC` or `festim.HenrysBC`
  whose `pressure` is the `GasSpecies`. The surface concentration is tied *instantaneously* to the gas
  pressure through the solubility law ($c = S\sqrt{P}$ for Sieverts, $c = K_H P$ for Henry). Use this
  when the surface is assumed to be at equilibrium with the gas.

This page covers the Dirichlet mode and verifies it against analytical solutions.

```{admonition} The pressure lives in a real function space
:class: note
Because the pressure is a single global unknown, it cannot be interpolated into a nodal field, so these
boundary conditions can only be enforced **weakly**, with Nitsche's method. Pass `enforce_weakly=True`
and a `penalty` coefficient.
```

+++

## A helper to build the slab + enclosure

We reuse the same setup for both verifications: a 1D slab whose left surface is coupled to a gas
enclosure by Henry's law, optionally with a perfect sink ($c=0$) on the right surface.

```{code-cell} ipython3
import numpy as np
import festim as F


def build(length, area, volume, D_0=1e-1, T=500.0, P0=1e5, K_H=1e15,
          penalty=100, sink=False, final_time=4000.0, dt=20.0, n=60):
    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = F.Mesh1D(np.linspace(0.0, length, num=n + 1))

    material = F.Material(name="mat", D_0=D_0, E_D=0.0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0.0, length], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=length)
    model.subdomains = [vol, left, right]

    H = F.Species("H", subdomains=[vol])
    model.species = [H]
    model.temperature = T

    H2 = F.GasSpecies(name="H2", initial_pressure=P0)
    model.enclosures = [
        F.Enclosure(volume=volume, species=[H2], temperature=T, surfaces={left: area})
    ]

    # Henry's law coupling: the surface concentration follows c = K_H * P(t)
    model.boundary_conditions = [
        F.HenrysBC(subdomain=left, H_0=K_H, E_H=0.0, pressure=H2, species=H,
                   enforce_weakly=True, penalty=penalty)
    ]
    if sink:
        model.boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=right, value=0.0, species=H)
        )

    model.initial_conditions = [F.InitialConcentration(value=0.0, species=H, volume=vol)]

    pressure = F.GasPressure(field=H2)
    model.exports = [pressure]

    # atol must match the magnitude of the residual: with c ~ K_H*P ~ 1e20 the residual
    # bottoms out well above the 1e-8 that would suit an order-1 problem.
    model.settings = F.Settings(atol=1e8, rtol=1e-8, transient=True,
                                final_time=final_time, stepsize=F.Stepsize(dt))
    model.show_progress_bar = False
    return model, H2, pressure
```

## Verification 1 — a closed enclosure reaches equilibrium

With no sink, the enclosure and the slab share a fixed number of hydrogen atoms. At equilibrium the
concentration is uniform and in equilibrium with the gas ($c = K_H P_\infty$ everywhere), so conservation
of atoms gives

$$
A\,\ell\,K_H\,P_\infty + \frac{P_\infty V}{k_B T} = \frac{P_0 V}{k_B T}
\qquad\Longrightarrow\qquad
P_\infty = \frac{P_0}{1 + A\,\ell\,K_H\,k_B T / V}.
$$

```{code-cell} ipython3
length, area, V_enc, T, P0, K_H = 2.0, 0.25, 1e-6, 500.0, 1e5, 1e15

model, H2, pressure = build(length, area, V_enc, T=T, P0=P0, K_H=K_H)
model.initialise()
model.run()

kT = F.k_B_SI * T
expected = P0 / (1 + area * length * K_H * kT / V_enc)

print(f"final pressure : {H2.value:.6e} Pa")
print(f"analytical     : {expected:.6e} Pa")
assert abs(H2.value - expected) / expected < 1e-4
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

plt.plot(pressure.t, pressure.data, label="FESTIM")
plt.axhline(expected, color="k", ls="--", label="analytical equilibrium")
plt.xlabel("time (s)")
plt.ylabel("H2 pressure (Pa)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Verification 2 — the TMAP permeation case (issue #996)

Now add a perfect sink on the far side. Hydrogen coupled in through Henry's law on the left permeates
across the slab and is pumped away on the right, so the enclosure pressure decays. Separating variables
gives the eigenvalue equation

$$
\lambda \tan\lambda = \beta, \qquad \beta = \frac{k_B T\,K_H\,A\,\ell}{V},
$$

and at late times the pressure decays as $\exp(-D\,\lambda_1^2\,t/\ell^2)$, where $\lambda_1$ is the
first root.

```{code-cell} ipython3
import scipy.optimize

D_0 = 1e-1
beta = F.k_B_SI * T * K_H * area * length / V_enc
lambda_1 = scipy.optimize.brentq(
    lambda lam: lam * np.tan(lam) - beta, 1e-12, np.pi / 2 - 1e-9
)
continuous_rate = D_0 * lambda_1**2 / length**2
print(f"beta = {beta:.3f},  lambda_1 = {lambda_1:.4f},  decay rate = {continuous_rate:.3e} 1/s")
```

```{code-cell} ipython3
model, H2, pressure = build(length, area, V_enc, D_0=D_0, T=T, P0=P0, K_H=K_H,
                            sink=True, final_time=300.0, dt=2.0)
model.initialise()
model.run()

t = np.array(pressure.t)
P = np.array(pressure.data)

# fit the rate where the first eigenmode dominates (higher modes gone, before the solve
# stalls on tolerances)
window = (P < 1e-2 * P[0]) & (P > 1e-6 * P[0])
fitted_rate = -np.polyfit(t[window], np.log(P[window]), 1)[0]

# backward Euler integrates dP/dt = -r P as P_{n+1} = P_n/(1 + r*dt), a log-slope of
# -ln(1 + r*dt)/dt; compare against that discrete rate rather than the continuous one.
dt = 2.0
discrete_rate = np.log(1 + continuous_rate * dt) / dt
print(f"fitted rate   : {fitted_rate:.4e} 1/s")
print(f"discrete rate : {discrete_rate:.4e} 1/s")
assert abs(fitted_rate - discrete_rate) / discrete_rate < 5e-3
```

```{code-cell} ipython3
:tags: [hide-input]

plt.semilogy(t, P, label="FESTIM")
plt.semilogy(t, P[0] * np.exp(-fitted_rate * t), "k--", label="fitted first mode")
plt.xlabel("time (s)")
plt.ylabel("H2 pressure (Pa)")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.show()
```
