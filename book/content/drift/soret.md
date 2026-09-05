---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

# Soret effect

The Soret effect (thermodiffusion, or thermophoresis) is the drift of hydrogen along a temperature
gradient. {py:class}`festim.SoretTerm` adds the corresponding term to the flux:

$$ J = -D \nabla c - D \frac{Q^* c}{k_B T^2} \nabla T $$

where $Q^*$ is the **heat of transport**, in eV.

Objectives:
* Adding a `SoretTerm` to a model
* Seeing where hydrogen ends up, and why the sign of $Q^*$ decides it
* Checking the result against the analytical steady state

+++

## A closed slab in a temperature gradient

The cleanest way to see the effect on its own is to take a slab that hydrogen cannot leave, fill it
uniformly, hold a temperature gradient across it and let it redistribute.

We use a 1 mm slab held at 500 K on the left and 1000 K on the right.

```{code-cell} ipython3
import numpy as np
import festim as F

L = 1e-3  # m
T_cold, T_hot = 500.0, 1000.0  # K
Q_star = 0.2  # eV


def temperature(x):
    return T_cold + (T_hot - T_cold) * x[0] / L
```

```{important}
The Soret term needs a temperature that varies **in space**. Writing
`my_model.temperature = lambda t: ...` gives a temperature that is uniform in space however much it
varies in time, and $\nabla T$ is then zero. FESTIM warns and drops the term when that happens.

For a temperature that comes out of a heat transfer solve rather than a formula, couple the two
problems with {py:class}`festim.CoupledTransientHeatTransferHydrogenTransport` — see
[](../temperatures/temperatures_advanced.md).
```

The model below takes `Q_star=None` to switch the drift off, so we can run the same case with and
without it. Note that there are **no boundary conditions at all**: an untagged boundary carries the
natural condition of zero total flux, which is exactly what an impermeable wall is.

```{code-cell} ipython3
def run(Q_star=None):
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(np.linspace(0, L, 200))

    material = F.Material(D_0=1e-7, E_D=0.2)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material)
    my_model.subdomains = [vol]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = temperature
    my_model.initial_conditions = [
        F.InitialConcentration(value=1.0, species=H, volume=vol)
    ]

    if Q_star is not None:
        my_model.drift_terms = [F.SoretTerm(species=H, Q_star=Q_star, subdomain=vol)]

    profile = F.Profile1DExport(field=H, subdomain=vol)
    inventory = F.TotalVolume(field=H, volume=vol)
    my_model.exports = [profile, inventory]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-10, final_time=5000)
    my_model.settings.stepsize = F.Stepsize(
        50, growth_factor=1.2, target_nb_iterations=4
    )

    my_model.initialise()
    my_model.run()
    return profile, inventory


profile_off, inventory_off = run(Q_star=None)
profile_on, inventory_on = run(Q_star=Q_star)
```

Without the drift term nothing happens — the slab was already uniform and there is nothing to drive
a flux. With it, hydrogen piles up at the **cold** end:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(profile_off.x * 1e3, profile_off.data[-1], "--", label="no Soret term")
ax.plot(profile_on.x * 1e3, profile_on.data[-1], label=f"$Q^* = {Q_star}$ eV")
ax.set_xlabel("x (mm)")
ax.set_ylabel("c (normalised)")
ax.set_title(f"{T_cold:.0f} K on the left, {T_hot:.0f} K on the right")
ax.legend()
plt.show()
```

+++

## Where the hydrogen goes

At steady state the total flux vanishes everywhere, so

$$ \frac{\nabla c}{c} = -\frac{Q^*}{k_B T^2} \nabla T
   \qquad \Longrightarrow \qquad
   c \propto \exp\left(\frac{Q^*}{k_B T}\right) $$

A **positive** $Q^*$ therefore accumulates hydrogen where $T$ is low. Our profile should match that
exponential once it is scaled to hold the same amount of hydrogen:

```{code-cell} ipython3
x = profile_on.x
analytical = np.exp(Q_star / (F.k_B * temperature([x])))
analytical *= np.trapezoid(profile_on.data[-1], x) / np.trapezoid(analytical, x)

error = np.max(np.abs(profile_on.data[-1] - analytical) / analytical)
print(f"max relative error: {error:.2e}")
assert error < 1e-4
```

Because the term is assembled in divergence form, the slab holds exactly as much hydrogen at the
end as it did at the start:

```{code-cell} ipython3
drift = abs(inventory_on.data[-1] - inventory_on.data[0]) / inventory_on.data[0]
print(f"inventory change: {drift:.2e} (relative)")
assert drift < 1e-10
```

+++

## Reversing the drift

$Q^*$ is a material property and can have either sign. A negative $Q^*$ sends hydrogen to the
**hot** end instead:

```{code-cell} ipython3
profile_neg, _ = run(Q_star=-Q_star)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(profile_on.x * 1e3, profile_on.data[-1], label=f"$Q^* = {Q_star}$ eV")
ax.plot(profile_neg.x * 1e3, profile_neg.data[-1], label=f"$Q^* = {-Q_star}$ eV")
ax.axhline(1.0, color="grey", lw=0.8, ls=":", label="initial")
ax.set_xlabel("x (mm)")
ax.set_ylabel("c (normalised)")
ax.legend()
plt.show()
```

The ratio between the two ends depends only on $Q^*$ and the two temperatures:

```{code-cell} ipython3
expected = np.exp(Q_star / F.k_B * (1 / T_cold - 1 / T_hot))
measured = profile_on.data[-1][0] / profile_on.data[-1][-1]
print(f"c(cold) / c(hot) = {measured:.3f}, expected {expected:.3f}")
```

+++

## Other drift terms

`SoretTerm` sits alongside {py:class}`festim.ElectromigrationTerm` and
{py:class}`festim.AdvectionTerm` in `drift_terms`, and several may act on the same species at once
— their velocities add. See [](electromigration.md) and [](advection.md).
