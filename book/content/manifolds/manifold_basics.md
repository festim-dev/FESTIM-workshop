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

# Transport along a manifold

Objectives:
* Declaring a codimension-1 subdomain
* Coupling it to the bulk, and getting the units right
* Boundary conditions and trapping on a manifold
* Getting derived quantities out of one

+++

## Declaring a manifold

A manifold is an ordinary {py:class}`festim.VolumeSubdomain` with one extra argument: `dim`, set to
one less than the dimension of the mesh.

```{code-cell} ipython3
import numpy as np
import festim as F

grain_boundary = F.VolumeSubdomain(
    id=2,
    material=F.Material(D_0=1e-1, E_D=0.0),
    dim=1,  # a line inside a 2D mesh
    locator=lambda x: np.isclose(x[0], 0.5),
)
```

Two things follow from that one argument:

* the subdomain is tagged in the **facet** meshtags rather than the cell meshtags, so its `id` has
  to be unique among the *surface* subdomains as well as the volume ones;
* it can be used directly wherever a surface is expected — as the `subdomain` of a
  {py:class}`festim.ParticleFluxBC`, or the `surface` of a {py:class}`festim.SurfaceFlux`. There is
  no need to declare a separate {py:class}`festim.SurfaceSubdomain` on the same facets.

```{important}
Manifolds are only supported by {py:class}`festim.HydrogenTransportProblemDiscontinuous`, because
each subdomain needs its own submesh and its own field.
```

A manifold may sit on the outer boundary of the mesh, or *inside* it. An interior one exchanges with
every volume subdomain it touches: one when it is buried inside a single material (the case below),
two when it separates a pair of them, and as many as there are grains for a boundary network
threading a polycrystal.

+++

## A fast path through a slow bulk

Here is the case a manifold exists for. A square of slow material conducts hydrogen from a loaded
bottom edge to an empty top edge. Down the middle of it runs a grain boundary, along which hydrogen
moves far more easily than through the grain interiors.

```{code-cell} ipython3
import dolfinx
from mpi4py import MPI

D_bulk = 1e-3  # m2/s, through the grain
D_gb = 1e-1  # m2/s, along the grain boundary
k_exchange = 1e-1  # m/s, grain <-> grain boundary
```

The bulk and the manifold each get their own {py:class}`festim.Species`, tied to their subdomain
through `subdomains`. The exchange between them is written **twice**: once as a flux leaving the
bulk, once as a source entering the manifold. `species_dependent_value` is what lets each half see
both concentrations, even though they live on different meshes.

```{code-cell} ipython3
def build(n=60):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)

    bulk = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=D_bulk, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    gb = F.VolumeSubdomain(
        id=2,
        material=F.Material(D_0=D_gb, E_D=0.0),
        dim=1,
        locator=lambda x: np.isclose(x[0], 0.5),
    )
    bottom = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[1], 0.0))
    top = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[1], 1.0))

    H_bulk = F.Species("H_bulk", subdomains=[bulk])
    H_gb = F.Species("H_gb", subdomains=[gb])

    exchange_bc = F.ParticleFluxBC(
        subdomain=gb,
        species=H_bulk,
        value=lambda c_bulk, c_gb: -k_exchange * (c_bulk - c_gb),
        species_dependent_value={"c_bulk": H_bulk, "c_gb": H_gb},
    )
    exchange_source = F.ParticleSource(
        volume=gb,
        species=H_gb,
        value=lambda c_bulk, c_gb: k_exchange * (c_bulk - c_gb),
        species_dependent_value={"c_bulk": H_bulk, "c_gb": H_gb},
    )

    bcs = [
        F.FixedConcentrationBC(subdomain=bottom, value=1.0, species=H_bulk),
        F.FixedConcentrationBC(subdomain=top, value=0.0, species=H_bulk),
        exchange_bc,
    ]
    return {
        "mesh": mesh,
        "bulk": bulk,
        "gb": gb,
        "bottom": bottom,
        "top": top,
        "H_bulk": H_bulk,
        "H_gb": H_gb,
        "bcs": bcs,
        "sources": [exchange_source],
    }
```

```{warning}
**Mind the units.** A {py:class}`festim.ParticleFluxBC` value is a flux, H/m²/s, while a
{py:class}`festim.ParticleSource` value is a volumetric rate over the subdomain it applies to.
Writing the same expression on both sides, as above, is only consistent for one choice of units on
the manifold species.

FESTIM imposes no convention. A line in a 2D mesh is a plane seen edge-on, so the natural reading
here is an **areal density**, H/m², and the same $J$ then appears on both sides. If instead you want
the manifold species to be a volumetric concentration inside a layer of thickness $\lambda$ — H/m³,
which is what you need if the layer has a solubility you want to compare with the bulk — the source
becomes $J/\lambda$. Keeping the problem dimensionally consistent is up to you.
```

The manifold has ends of its own: the two points where the line meets the bottom and top edges of the
mesh. The loading surface loads the mouth of the grain boundary as well as the grain, so we put a
concentration on the bottom end; that is covered in more detail
[below](boundary-conditions-at-the-ends-of-a-manifold).

```{code-cell} ipython3
def run(with_gb):
    p = build()
    subdomains = [p["bulk"], p["bottom"], p["top"]]
    species = [p["H_bulk"]]
    bcs = list(p["bcs"])
    sources = []
    exports = [F.SurfaceFlux(field=p["H_bulk"], surface=p["top"])]

    if with_gb:
        gb_mouth = F.SurfaceSubdomain(
            id=5, dim=0, locator=lambda x: np.isclose(x[1], 0.0)
        )
        subdomains += [p["gb"], gb_mouth]
        species.append(p["H_gb"])
        sources = p["sources"]
        bcs.append(
            F.FixedConcentrationBC(subdomain=gb_mouth, value=1.0, species=p["H_gb"])
        )
        exports += [
            F.TotalVolume(field=p["H_gb"], volume=p["gb"]),
            F.AverageVolume(field=p["H_gb"], volume=p["gb"]),
            F.SurfaceFlux(field=p["H_bulk"], surface=p["gb"]),
            F.SurfaceFlux(field=p["H_gb"], surface=gb_mouth),
        ]
    else:
        bcs.remove(p["bcs"][-1])  # no manifold, no exchange

    my_model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(p["mesh"]),
        subdomains=subdomains,
        species=species,
        sources=sources,
        boundary_conditions=bcs,
        exports=exports,
        temperature=500,
        settings=F.Settings(atol=1e-12, rtol=1e-10, transient=False),
    )
    my_model.initialise()
    my_model.run()
    return my_model, p


model_plain, parts_plain = run(with_gb=False)
model_gb, parts_gb = run(with_gb=True)
```

```{code-cell} ipython3
flux_plain = model_plain.exports[0].data[-1]
flux_gb = model_gb.exports[0].data[-1]

print(f"flux through the top surface, no grain boundary: {flux_plain:.3e}")
print(f"flux through the top surface, with it          : {flux_gb:.3e}")
print(f"ratio: {flux_gb / flux_plain:.2f}")
```

A single line of mesh, with no cells of its own, carries several times what the whole square does.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

c_plain = parts_plain["H_bulk"].subdomain_to_post_processing_solution[
    parts_plain["bulk"]
]
c_gb_bulk = parts_gb["H_bulk"].subdomain_to_post_processing_solution[parts_gb["bulk"]]

fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, u, title in zip(
    axs, [c_plain, c_gb_bulk], ["no grain boundary", "with grain boundary"]
):
    coords = u.function_space.tabulate_dof_coordinates()
    cs = ax.tricontourf(
        coords[:, 0], coords[:, 1], u.x.array, levels=np.linspace(0, 1, 41)
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
fig.colorbar(cs, ax=axs, label="c in the grain")
axs[0].set_ylabel("y (m)")
plt.show()
```

The contours bulge upwards around $x = 0.5$. The loaded surface feeds the mouth of the grain
boundary directly; the grain boundary carries that hydrogen up the square with barely any drop in
concentration, and leaks it sideways into the grain all the way along. The grain interiors then only
have to carry it the short distance from the boundary to the top edge.

+++

(manifolds-several-subdomains)=
## Manifolds between several subdomains

The grain boundary above is buried inside a single material, so one exchange pair describes it.
When the manifold *separates* volume subdomains, declare **one exchange per side** — one
{py:class}`festim.ParticleFluxBC` and one {py:class}`festim.ParticleSource` each. Both name the same
manifold; FESTIM works out which side each belongs to from the bulk species it reads:

```python
for bulk_species, k in zip(grain_species, exchange_rates):
    bcs.append(
        F.ParticleFluxBC(
            subdomain=gamma,
            species=bulk_species,
            value=lambda c_bulk, c_man, k=k: -k * (c_bulk - c_man),
            species_dependent_value={"c_bulk": bulk_species, "c_man": H_manifold},
        )
    )
    sources.append(
        F.ParticleSource(
            volume=gamma,
            species=H_manifold,
            value=lambda c_bulk, c_man, k=k: k * (c_bulk - c_man),
            species_dependent_value={"c_bulk": bulk_species, "c_man": H_manifold},
        )
    )
```

A single source may **not** read the bulk concentrations of several sides at once: an interior
manifold is integrated over interior facets, where each term has to be restricted to one side.

The pattern scales past two subdomains, which is what a grain-boundary network in a polycrystal
needs: declare the whole network as **one** manifold, so it carries a single connected field and
hydrogen crosses triple junctions with no junction condition to write; give each grain its own
`VolumeSubdomain` and species; and loop the two objects above over the grains. The exchange law
belongs to the (manifold, grain) pair, so a boundary that blocks one grain while conducting into
another is written by giving that grain a small rate.

```{note}
A pair of volume subdomains may be separated either by a {py:class}`festim.Interface` — imposing a
jump in concentration across a shared boundary, see [](../material/material_basics.md) — or by a
manifold carrying its own transport equation, but not both. FESTIM raises if an interface and a
manifold cover the same facets.
```

+++

(boundary-conditions-at-the-ends-of-a-manifold)=
## Boundary conditions at the ends of a manifold

A manifold has a boundary of its own: the endpoints of a line in a 2D mesh, the rim of a surface in
a 3D mesh. Declare it as a {py:class}`festim.SurfaceSubdomain` with `dim` set to the mesh dimension
minus **two** — just as a manifold is a `VolumeSubdomain` with `dim` set to the mesh dimension minus
one:

```{code-cell} ipython3
gb_mouth = F.SurfaceSubdomain(id=5, dim=0, locator=lambda x: np.isclose(x[1], 0.0))
```

The locator is evaluated **on the manifold**, not on the parent mesh, and must select a point on its
boundary — a locator matching only interior points raises rather than silently doing nothing.

Such a surface carries no meshtag, so its `id` does not have to differ from a manifold or interface
id. Which manifold it bounds is taken from the `species` of the boundary condition using it, so that
species must live on exactly one manifold; the same surface object can be reused on several
manifolds, one species each.

Without a condition of this kind, the ends of a manifold carry the natural zero-flux condition —
which is why the top end of the grain boundary above needs nothing written on it.

```{note}
Boundary conditions on the boundary of a manifold are limited to
{py:class}`festim.FixedConcentrationBC` and {py:class}`festim.OutflowBC`.
```

+++

## Reactions and trapping

A reaction runs on a manifold like on any other volume subdomain: give it `volume=gamma` and species
that live there. Trapping is written as a reaction against {py:class}`festim.ImplicitSpecies` empty
sites, exactly as in [](../species_reactions/reactions.ipynb):

```{code-cell} ipython3
gb = parts_gb["gb"]
H_gb = parts_gb["H_gb"]

trapped = F.Species("trapped", mobile=False, subdomains=[gb])
empty_sites = F.ImplicitSpecies(n=1.0, others=[trapped], name="empty_sites")

trapping = F.ArrheniusReaction(
    reactant=[H_gb, empty_sites],
    product=trapped,
    k_0=1e-1,
    E_k=0.0,
    p_0=1e-2,
    E_p=0.0,
    volume=gb,
)
print(trapping)
```

```{warning}
{py:class}`festim.Trap` is **not** a shortcut for this: it builds a species without `subdomains`, so
the trapped species has to be declared by hand as above.

The density `n` of an implicit species consumed on a manifold is a coefficient of an integral over
that manifold, so FESTIM builds it there. Two consequences: give `n` as a float or as a callable of
`x` and `t` rather than as a ready-made `dolfinx.fem.Function`, which cannot be moved; and declare
one implicit species per subdomain rather than sharing one between a reaction on a manifold and a
reaction elsewhere. Both are raised rather than silently mis-assembled.
```

+++

## Derived quantities

Once a manifold is in the mesh, quantities can be asked for in three places. All three are already
attached to the model we ran above:

**Over the manifold**, for its own species: a *volume* quantity with `volume` set to the manifold.
It is integrated over the manifold itself, so a {py:class}`festim.TotalVolume` on a line in a 2D mesh
is a line integral.

**On the facets the manifold occupies**, for a *bulk* species: a *surface* quantity, with the
manifold passed where a surface subdomain normally goes. This is the net exchange between the bulk
and the manifold.

**On the boundary of the manifold**: a surface quantity whose `surface` is the codimension-2
subdomain from the previous section.

```{code-cell} ipython3
_, inventory, average, exchange, mouth = model_gb.exports

print(f"inventory along the grain boundary : {inventory.data[-1]:.3e}")
print(f"average concentration in it        : {average.data[-1]:.3e}")
print(f"net exchange with the grain        : {exchange.data[-1]:+.3e}")
print(f"flux in through its loaded end     : {mouth.data[-1]:+.3e}")
```

The sign convention is that a positive flux *leaves* the subdomain the species lives on. `exchange`
reads the bulk species, so its negative value means hydrogen is going **into** the grain: over the
whole line the grain boundary is a net donor, which is the shortcut at work. Both figures are
negative for the same reason — the boundary takes hydrogen in at its loaded end and gives it back to
the grain higher up.

Note that the two are not the same kind of number. `exchange` is integrated over the length of the
manifold, while a quantity on a codimension-2 boundary is evaluated at a point, so their units
differ and they are not meant to be added up.

```{code-cell} ipython3
:tags: [hide-input]

c_gb = H_gb.subdomain_to_post_processing_solution[gb]
coords = c_gb.function_space.tabulate_dof_coordinates()
order = np.argsort(coords[:, 1])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(coords[order, 1], c_gb.x.array[order])
ax.set_xlabel("y (m)")
ax.set_ylabel("c along the grain boundary")
plt.show()
```

The profile confirms it: the concentration along the boundary hardly falls below the value imposed
at its mouth, because it moves along the manifold far faster than it leaks out sideways.

On an interior manifold, which side a surface quantity is read on follows from `field`, exactly as
it does for the flux boundary conditions above: declare one export per side, each naming that side's
species.

Asking for a manifold's own species on its own facets raises: it has no flux across itself, and the
quantity meant is the volume one.

+++

## Limitations

```{note}
* Only codimension 1 is supported. A codimension-2 subdomain carrying its own equation is not: a
  bulk field has no well-defined trace on a line in 3D or a point in 2D, so the exchange with it
  would not be well posed.
* A manifold must lie wholly inside the mesh or wholly on its boundary, and one on the boundary of
  the mesh is adjacent to a single volume subdomain.
* Quantities on and around a manifold are limited to the integral-based ones
  ({py:class}`festim.SurfaceFlux`, {py:class}`festim.TotalSurface`,
  {py:class}`festim.AverageSurface`, {py:class}`festim.TotalVolume`,
  {py:class}`festim.AverageVolume`) and field exports. The minimum and maximum quantities and
  {py:class}`festim.CustomQuantity` are not available in
  {py:class}`festim.HydrogenTransportProblemDiscontinuous` at all, manifold or not.
* Cartesian coordinates only.
```

Next: [](pipe_wall.md) puts a manifold to work as a coolant channel, with hydrogen advected along it
and leaving through an outlet.
