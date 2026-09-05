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

# Electromigration

A charged species in an electric field drifts along it. {py:class}`festim.ElectromigrationTerm`
adds the Nernst-Planck term to the flux:

$$ J = -D \nabla c - \frac{z D c}{k_B T} \nabla \varphi $$

with the potential $\varphi$ in volts and $z$ the **charge number** of the species ($+1$ for a
proton, $-1$ for an electron, $+2$ for an oxygen vacancy). Because $k_B$ is written in eV/K, it
already carries the elementary charge.

Objectives:
* Adding an `ElectromigrationTerm` to a model
* Seeing how the charge number sets the direction of the drift
* Checking the result against the Boltzmann distribution

+++

## A membrane under a bias

We take a 1 mm membrane at a uniform 500 K, with a 0.1 V bias applied across it, and let a
uniformly distributed charged species redistribute. As in [](soret.md) the membrane is closed —
no boundary conditions, so the natural zero-total-flux condition applies on both ends.

```{code-cell} ipython3
import numpy as np
import festim as F

L = 1e-3  # m
T = 500.0  # K
delta_phi = 0.1  # V


def potential(x):
    return delta_phi * (1 - x[0] / L)
```

```{note}
The potential is **prescribed**: FESTIM does not solve for it. Give it as a float, a callable of
`x`, `t` and/or `T`, or a ready-made fenics object — the same input conventions as everywhere else
in FESTIM. A spatially uniform potential makes $\nabla \varphi$ zero, so the term does nothing and
FESTIM warns.
```

```{code-cell} ipython3
def run(charge):
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(np.linspace(0, L, 200))

    material = F.Material(D_0=1e-7, E_D=0.2)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material)
    my_model.subdomains = [vol]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T
    my_model.initial_conditions = [
        F.InitialConcentration(value=1.0, species=H, volume=vol)
    ]

    if charge != 0:
        my_model.drift_terms = [
            F.ElectromigrationTerm(
                species=H, charge=charge, potential=potential, subdomain=vol
            )
        ]

    profile = F.Profile1DExport(field=H, subdomain=vol)
    my_model.exports = [profile]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-10, final_time=5000)
    my_model.settings.stepsize = F.Stepsize(
        50, growth_factor=1.2, target_nb_iterations=4
    )

    my_model.initialise()
    my_model.run()
    return profile
```

Running the same membrane for a negative, a neutral and a positive species:

```{code-cell} ipython3
profiles = {z: run(z) for z in (-1, 0, 1)}
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

fig, (ax_phi, ax_c) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

x = profiles[0].x
ax_phi.plot(x * 1e3, potential([x]), color="grey")
ax_phi.set_ylabel(r"$\varphi$ (V)")

for z, profile in profiles.items():
    ax_c.plot(profile.x * 1e3, profile.data[-1], label=f"$z = {z:+d}$")
ax_c.set_xlabel("x (mm)")
ax_c.set_ylabel("c (normalised)")
ax_c.legend()
plt.show()
```

Positively charged species drift **down** the potential gradient and pile up on the low-potential
side; negatively charged ones do the opposite; a neutral species ($z = 0$, or simply no drift term)
stays uniform.

+++

## Where the species goes

At steady state the total flux vanishes, so the profile is the Boltzmann distribution

$$ c \propto \exp\left(-\frac{z \varphi}{k_B T}\right) $$

```{code-cell} ipython3
profile = profiles[1]
x = profile.x

analytical = np.exp(-1 * potential([x]) / (F.k_B * T))
analytical *= np.trapezoid(profile.data[-1], x) / np.trapezoid(analytical, x)

error = np.max(np.abs(profile.data[-1] - analytical) / analytical)
print(f"max relative error: {error:.2e}")
assert error < 1e-3
```

so the ratio across the membrane depends only on the bias, the charge and the temperature:

```{code-cell} ipython3
for z, profile in profiles.items():
    measured = profile.data[-1][0] / profile.data[-1][-1]
    expected = np.exp(-z * delta_phi / (F.k_B * T))
    print(f"z = {z:+d}:  c(0)/c(L) = {measured:8.4f}   expected {expected:8.4f}")
```

```{tip}
The drift grows exponentially with $z \, \Delta\varphi / k_B T$. At 500 K, $k_B T$ is only 43 meV,
so a bias of a few tenths of a volt already separates the two ends by orders of magnitude — and a
profile that steep needs a fine enough mesh to resolve, see the note on the cell Péclet number in
[](advection.md).
```
