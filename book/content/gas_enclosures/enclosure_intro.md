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

# Filling an enclosure

In this first example a 1D slab is initially loaded with hydrogen. Its right surface faces a gas
enclosure that starts under vacuum. Hydrogen diffuses to that surface, recombines into H₂
($2\,\mathrm{H} \rightleftharpoons \mathrm{H_2}$) and fills the enclosure, so the partial pressure
builds up over time.

The pressure is a genuine unknown, solved together with the transport problem, so the surface reaction
sees the pressure it has itself created: as $P$ rises, dissociation ($k_d P$) pushes back against
recombination ($k_r c^2$) until the surface reaches equilibrium.

Objectives:
* Attaching a `festim.Enclosure` to a transport problem
* Coupling it to the solid through a `festim.SurfaceReactionBC`
* Exporting the pressure with `festim.GasPressure`
* Checking that hydrogen is conserved between the solid and the gas

+++

## Parameters

```{code-cell} ipython3
import numpy as np

L = 1e-3       # slab thickness (m)
AREA = 1e-2    # area of the membrane facing the enclosure (m2)
VOLUME = 1e-5  # volume of the enclosure (m3)
TEMPERATURE = 600.0  # K

c_0 = 1e21             # initial concentration in the slab (m-3)
D_0, E_D = 1e-7, 0.2   # diffusion coefficient

k_r0, E_kr = 1e-26, 0.0  # recombination: 2H -> H2
k_d0, E_kd = 1e14, 0.0   # dissociation: H2 -> 2H

final_time = 5000.0
stepsize = 50.0
```

## Building the model

Enclosures are only available on the `festim.HydrogenTransportProblemDiscontinuous` class (the
base `HydrogenTransportProblem` raises a `NotImplementedError` if given enclosures).

```{code-cell} ipython3
import festim as F

my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh1D(np.linspace(0, L, num=100))

material = F.Material(name="tungsten", D_0=D_0, E_D=E_D)
slab = F.VolumeSubdomain1D(id=1, borders=[0, L], material=material)
left = F.SurfaceSubdomain1D(id=1, x=0)
right = F.SurfaceSubdomain1D(id=2, x=L)
my_model.subdomains = [slab, left, right]

H = F.Species("H", subdomains=[slab])
my_model.species = [H]
my_model.temperature = TEMPERATURE
```

Now the gas side. We create a `festim.GasSpecies`, the species whose partial pressure is the
unknown, and put it in an `festim.Enclosure`. Because the slab is 1D, its right surface is a
point and carries no area, so we give the enclosure the membrane area explicitly.

```{code-cell} ipython3
H2 = F.GasSpecies(name="H2", initial_pressure=0.0)
enclosure = F.Enclosure(
    volume=VOLUME,
    species=[H2],
    temperature=TEMPERATURE,
    surfaces={right: AREA},
    name="plenum",
)
my_model.enclosures = [enclosure]
```

The reaction that couples the two sides is a `festim.SurfaceReactionBC`. Passing the
`GasSpecies` as `gas_pressure` (rather than a float) is what makes the pressure an unknown instead of an
imposed value:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.SurfaceReactionBC(
        reactant=[H, H],
        gas_pressure=H2,
        k_r0=k_r0,
        E_kr=E_kr,
        k_d0=k_d0,
        E_kd=E_kd,
        subdomain=right,
    )
]

my_model.initial_conditions = [
    F.InitialConcentration(value=c_0, species=H, volume=slab)
]
```

## Exports, settings and run

We export the enclosure pressure with `festim.GasPressure` and the amount of hydrogen left in
the slab with `festim.TotalVolume`.

```{code-cell} ipython3
pressure = F.GasPressure(field=H2)
inventory = F.TotalVolume(field=H, volume=slab)
my_model.exports = [pressure, inventory]

my_model.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    transient=True,
    final_time=final_time,
    stepsize=F.Stepsize(stepsize),
)

my_model.initialise()
my_model.run()
```

## The pressure build-up

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

t = np.array(pressure.t)
P = np.array(pressure.data)

plt.plot(t, P, color="tab:red")
plt.xlabel("time (s)")
plt.ylabel("H2 partial pressure (Pa)")
plt.title("Enclosure filling from a slab through 2H ⇌ H2")
plt.grid(alpha=0.3)
plt.show()
```

## Hydrogen is conserved

Nothing leaves the system, so every H atom lost by the slab ends up in the gas as half an H₂ molecule.
This is exact, not approximate: the same expression drives the surface flux and the pressure balance.
The number of H atoms in the gas is $2 P V / (k_B T)$ (two atoms per H₂ molecule).

```{code-cell} ipython3
:tags: [hide-input]

in_solid = AREA * np.array(inventory.data)
in_gas = 2.0 * P * VOLUME / (F.k_B_SI * TEMPERATURE)
total = in_solid + in_gas

fig, ax = plt.subplots()
ax.plot(t, in_solid, label="in the slab")
ax.plot(t, in_gas, label="in the enclosure")
ax.plot(t, total, "k--", label="total (conserved)")
ax.set_xlabel("time (s)")
ax.set_ylabel("H atoms")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

drift = abs(total[-1] - total[0]) / total[0]
print(f"final pressure   : {P[-1]:.4e} Pa")
print(f"inventory drift  : {drift:.2e} (relative)")
assert drift < 1e-6
```
