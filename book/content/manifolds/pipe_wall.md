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

# A coolant channel along a pipe wall

Hydrogen permeating through a pipe wall ends up in the coolant, which carries it away. Resolving the
coolant as a second fluid domain means meshing it, coupling it and solving it; if all you want is
how much hydrogen leaves with the coolant, a **manifold** running along the wall is enough.

This is the case the codimensional machinery was built for, and it uses most of it at once: a
manifold on the boundary of the mesh, an {py:class}`festim.AdvectionTerm` along it, and boundary
conditions on the manifold's own two ends.

Objectives:
* Advecting a species along a manifold
* Using `FixedConcentrationBC` and `OutflowBC` on the ends of one
* Checking the pickup rate against the permeation rate

+++

## Setup

A 1 m length of wall, 1 cm thick. Hydrogen is held at a fixed concentration on the inner surface
($y = 0$) and permeates outwards; the coolant channel is the line $y = H$ on the outer surface.
Clean coolant enters at $x = 0$ and flows along at 0.1 m/s.

```{code-cell} ipython3
import numpy as np
import dolfinx
from mpi4py import MPI
import festim as F

L = 1.0  # m, length of wall
H = 0.01  # m, wall thickness

D_wall = 1e-4  # m2/s
D_fluid = 1e-3  # m2/s, along the channel
k_exchange = 1.0  # m/s, wall surface <-> coolant
velocity = 0.1  # m/s
```

```{code-cell} ipython3
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [200, 10]
)
```

The wall is an ordinary volume subdomain; the channel is a manifold, `dim=1` in a 2D mesh, sitting on
the outer boundary of the mesh:

```{code-cell} ipython3
wall = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_wall, E_D=0.0),
    locator=lambda x: np.full_like(x[0], True, dtype=bool),
)
channel = F.VolumeSubdomain(
    id=2,
    material=F.Material(D_0=D_fluid, E_D=0.0),
    dim=1,
    locator=lambda x: np.isclose(x[1], H),
)
```

The inner surface of the wall is an ordinary {py:class}`festim.SurfaceSubdomain`. The two ends of the
channel are **codimension-2** surfaces: `dim=0` points in a 2D mesh, located on the manifold itself.

```{code-cell} ipython3
loaded_surface = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[1], 0.0))
inlet = F.SurfaceSubdomain(id=4, dim=0, locator=lambda x: np.isclose(x[0], 0.0))
outlet = F.SurfaceSubdomain(id=5, dim=0, locator=lambda x: np.isclose(x[0], L))
```

```{code-cell} ipython3
c_wall = F.Species("c_wall", subdomains=[wall])
c_fluid = F.Species("c_fluid", subdomains=[channel])
```

+++

## Advection along the manifold

An {py:class}`festim.AdvectionTerm` on a manifold takes an ordinary **ambient** velocity vector — two
components in a 2D mesh, three in a 3D one. There is no need to project it onto the manifold: the
tangential gradient is orthogonal to the normal, so $\mathbf{v} \cdot \nabla_\Gamma c$ ignores the
normal component of $\mathbf{v}$ by itself.

```{code-cell} ipython3
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
flow = dolfinx.fem.Function(V)
flow.interpolate(
    lambda x: np.vstack([np.full_like(x[0], velocity), np.zeros_like(x[0])])
)

advection = F.AdvectionTerm(velocity=flow, subdomain=channel, species=c_fluid)
```

+++

## Boundary conditions

Four conditions, two of them on the ends of the manifold:

```{code-cell} ipython3
boundary_conditions = [
    # the inner surface of the wall is loaded
    F.FixedConcentrationBC(subdomain=loaded_surface, value=1.0, species=c_wall),
    # the wall gives up hydrogen to the coolant wherever they touch
    F.ParticleFluxBC(
        subdomain=channel,
        species=c_wall,
        value=lambda c_w, c_f: -k_exchange * (c_w - c_f),
        species_dependent_value={"c_w": c_wall, "c_f": c_fluid},
    ),
    # clean coolant enters at one end of the channel ...
    F.FixedConcentrationBC(subdomain=inlet, value=0.0, species=c_fluid),
    # ... and leaves at the other
    F.OutflowBC(subdomain=outlet, species=c_fluid),
]

sources = [
    F.ParticleSource(
        volume=channel,
        species=c_fluid,
        value=lambda c_w, c_f: k_exchange * (c_w - c_f),
        species_dependent_value={"c_w": c_wall, "c_f": c_fluid},
    )
]
```

```{important}
The {py:class}`festim.OutflowBC` at the outlet is what makes this work. Drift terms are assembled in
divergence form, so an end with nothing written on it is a **closed** end: zero total flux, with the
flow balanced by back-diffusion. The coolant would arrive at the end of the pipe and stop. See
[](../drift/advection.md) for what that looks like when you forget.
```

+++

## Running it

```{code-cell} ipython3
permeation = F.SurfaceFlux(field=c_wall, surface=channel)
pickup = F.SurfaceFlux(field=c_fluid, surface=outlet)

my_model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(mesh),
    subdomains=[wall, channel, loaded_surface, inlet, outlet],
    species=[c_wall, c_fluid],
    sources=sources,
    boundary_conditions=boundary_conditions,
    drift_terms=[advection],
    exports=[permeation, pickup],
    temperature=500,
    settings=F.Settings(atol=1e-14, rtol=1e-12, transient=False),
)
my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

u_wall = c_wall.subdomain_to_post_processing_solution[wall]
u_fluid = c_fluid.subdomain_to_post_processing_solution[channel]

coords_fluid = u_fluid.function_space.tabulate_dof_coordinates()
order = np.argsort(coords_fluid[:, 0])

fig, (ax_wall, ax_fluid) = plt.subplots(
    2, 1, figsize=(7, 5), sharex=True, height_ratios=[1, 2]
)

coords = u_wall.function_space.tabulate_dof_coordinates()
cs = ax_wall.tricontourf(coords[:, 0], coords[:, 1], u_wall.x.array, levels=40)
fig.colorbar(cs, ax=ax_wall, label="c in the wall")
ax_wall.set_ylabel("y (m)")
ax_wall.set_title("wall, loaded from below")

ax_fluid.plot(coords_fluid[order, 0], u_fluid.x.array[order])
ax_fluid.set_xlabel("x (m)")
ax_fluid.set_ylabel("c in the coolant")
ax_fluid.set_title("coolant channel")
plt.tight_layout()
plt.show()
```

The coolant enters clean and picks hydrogen up all the way along, so its concentration climbs
towards the outlet. It is close to linear because the wall delivers hydrogen at almost the same rate
everywhere: the coolant never gets loaded enough to slow the permeation down.

+++

## Checking the pickup rate

At steady state whatever the wall gives up has to leave with the coolant, so the flux across the
manifold and the flux out of its end should agree:

```{code-cell} ipython3
print(f"permeation from the wall : {permeation.data[-1]:.6e}")
print(f"leaving at the outlet    : {pickup.data[-1]:.6e}")

imbalance = abs(permeation.data[-1] - pickup.data[-1]) / permeation.data[-1]
print(f"imbalance                : {imbalance:.2%}")
assert imbalance < 0.05
```

The few percent left over is discretisation error, and it shrinks as the mesh is refined.

This is also a direct check that {py:class}`festim.SurfaceFlux` reports the **total** flux on the
boundary of a manifold, advection included, just as it does in the bulk. Almost all of the pickup is
advective — the coolant is simply carrying the hydrogen out — and the diffusive part is a fraction of
a percent of it, with the opposite sign:

```{code-cell} ipython3
c_out = u_fluid.x.array[order][-1]
advective = velocity * c_out

print(f"outlet concentration      : {c_out:.6e}")
print(f"advective part, v * c_out : {advective:.6e}")
print(f"reported by SurfaceFlux   : {pickup.data[-1]:.6e}")
print(f"diffusive remainder       : {pickup.data[-1] - advective:.3e}")

assert np.isclose(pickup.data[-1], advective, rtol=1e-2)
```

+++

## Resolving the channel

The coolant is advection-dominated, so the mesh along the channel has to keep the **cell Péclet
number** in hand — see [](../drift/advection.md):

```{code-cell} ipython3
h = L / 200
print(f"cell Peclet number along the channel: {velocity * h / D_fluid:.2f}")
```

Note that this is set by the mesh along the manifold, which is the mesh of the parent domain. A wall
that needs few cells through its thickness may still need many along its length purely to resolve
what is happening in the channel.
