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

(gas-enclosures)=
# Gas enclosures

```{note}
Gas enclosures were introduced in FESTIM 2.2. They require `dolfinx >= 0.11` and the
`festim.HydrogenTransportProblemDiscontinuous` class.
```

FESTIM has always been able to impose a gas pressure on a surface — via `SievertsBC.pressure`,
`HenrysBC.pressure` or `SurfaceReactionBC.gas_pressure` — but only as an **input**: a number (or a
function of time) that you prescribe. There was no way to let the pressure of a gas volume *evolve* in
response to what permeates into or out of it.

A `festim.Enclosure` fixes that. It is a **0D gas volume** whose partial pressure is a genuine
unknown of the problem, solved **monolithically** with the transport problem — not updated in a separate,
staggered step between timesteps. Under the hood the pressure lives in a "real" function space (a single
global degree of freedom, native to dolfinx 0.11), so the pressure balance goes directly into the
variational form and the Newton solver sees the full coupling between the gas and the solid.

## The governing equation

For each gas species in an enclosure of volume $V$ and temperature $T$, the partial pressure $P$ evolves as

$$
\frac{dP}{dt} = \frac{k_B T}{V} \left( \sum_\Gamma A_\Gamma \int_\Gamma \varphi \, dS
                                        + \sum_\text{openings} Q \right)
$$

where $\varphi$ is the rate of particles entering the gas from the solid across a contact surface
$\Gamma$, $A_\Gamma$ the physical area of that surface, and $Q$ the molar flow rate through any
[openings](openings.md) (pumps, reservoirs, …).

## What this part covers

- [](enclosure_intro.md) — the basics: a slab filling a plenum, with the pressure as an unknown, coupled
  through a surface reaction ($2\,\mathrm{H} \rightleftharpoons \mathrm{H_2}$).
- [](coupling.md) — the two ways an enclosure couples to the solid: a **flux** coupling
  (`SurfaceReactionBC`) and a **Dirichlet / concentration** coupling (`SievertsBC` / `HenrysBC`), verified
  against the TMAP analytical solution.
- [](openings.md) — letting gas in or out: `festim.Pump`, `festim.Reservoir`,
  `festim.PrescribedFlowRate` and `festim.EnclosureConnection`.
- [](geometries.md) — enclosures on discontiguous 1D meshes and on 2D geometries.

## The building blocks

An enclosure holds one or more `festim.GasSpecies`, is given a volume and temperature, and lists
the surfaces it is in contact with (mapped to their physical area):

```{code-cell} ipython3
import festim as F

H2 = F.GasSpecies(name="H2", initial_pressure=0.0)
enclosure = F.Enclosure(
    volume=1e-5,       # m3
    species=[H2],
    temperature=600.0, # K, independent of the transport temperature
    name="plenum",
)
enclosure
```

```{admonition} Surface areas depend on the dimension
:class: important
The area turns a flux through a surface into a number of particles per second. How you provide it
depends on the mesh dimension:

- **1D**: a surface is a point and carries no extent — pass the membrane area facing the enclosure (m²).
- **2D**: a surface is a line — pass the out-of-plane depth of the model (m).
- **3D**: the mesh already measures the area — pass `1.0` (a plain list of surfaces is also accepted).
```
