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

# Advection and outflow

{py:class}`festim.AdvectionTerm` carries hydrogen with a moving fluid: the drift velocity is simply
the fluid velocity.

$$ J = -D \nabla c + c \, \mathbf{v} $$

This page is also where the **divergence form** is explained, because it changes what an untagged
boundary means for *every* drift term, not just this one.

Objectives:
* Adding an `AdvectionTerm` to a model
* Understanding why an outlet needs an `OutflowBC`
* Checking that `SurfaceFlux` reports the advected flux

```{warning}
Before FESTIM 2.2, `AdvectionTerm` was assembled as $\mathbf{v} \cdot \nabla c$ and lived in
`my_model.advection_terms`. It is now assembled as $\nabla \cdot (c \mathbf{v})$ and lives in
`my_model.drift_terms`. Existing models are unaffected **only if** the velocity field is
divergence-free *and* every boundary the flow crosses already carries a boundary condition. If it
does not, read on.
```

+++

## A channel with a flow through it

A 2 m by 1 m channel, with a parabolic (Poiseuille) velocity profile pushing hydrogen from left to
right. Hydrogen enters at the inlet and is absorbed by the top and bottom walls.

```{code-cell} ipython3
import numpy as np
import dolfinx
from mpi4py import MPI
import festim as F

L_x, L_y = 2.0, 1.0
v_max = 1.0  # m/s
D = 0.05  # m2/s
```

The velocity field is an ordinary `dolfinx.fem.Function` on a **vector** function space. In a real
model it would come out of a Navier-Stokes solve — see [](../applications/multiphysics/cfd.md) for
a velocity field imported from OpenFOAM — but here we interpolate it directly:

```{code-cell} ipython3
def make_velocity(mesh):
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    velocity = dolfinx.fem.Function(V)
    velocity.interpolate(
        lambda x: np.vstack(
            [v_max * 4 * x[1] * (L_y - x[1]) / L_y**2, np.zeros_like(x[0])]
        )
    )
    return velocity
```

```{code-cell} ipython3
def run(outflow_bc, nx=160, ny=80):
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [L_x, L_y]], [nx, ny]
    )

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh)

    vol = F.VolumeSubdomain(id=1, material=F.Material(D_0=D, E_D=0))
    inlet = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 0.0))
    outlet = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], L_x))
    walls = F.SurfaceSubdomain(
        id=4, locator=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], L_y)
    )
    my_model.subdomains = [vol, inlet, outlet, walls]

    H = F.Species("H")
    my_model.species = [H]

    velocity = make_velocity(mesh)
    my_model.drift_terms = [
        F.AdvectionTerm(velocity=velocity, subdomain=vol, species=H)
    ]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=inlet, value=1.0, species=H),
        F.FixedConcentrationBC(subdomain=walls, value=0.0, species=H),
    ]
    if outflow_bc:
        my_model.boundary_conditions.append(F.OutflowBC(subdomain=outlet, species=H))

    my_model.temperature = 500
    my_model.settings = F.Settings(atol=1e-12, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()
    return H.post_processing_solution
```

+++

## What an untagged boundary means now

The divergence form leaves behind the natural boundary condition on the **total** flux, drift
included. A boundary with nothing on it is therefore a wall: zero total flux, with the drift exactly
balanced by back-diffusion. That is right at a real wall, and wrong at an outlet — the flow arrives
at a closed end and the hydrogen backs up against it.

Running the channel with nothing on the outlet shows exactly that:

```{code-cell} ipython3
c_closed = run(outflow_bc=False)
print(f"peak concentration: {c_closed.x.array.max():.1f} times the inlet value")
```

{py:class}`festim.OutflowBC` marks a surface the flow leaves through. It cancels the drift boundary
term, so the natural condition there becomes zero *diffusive* flux — the standard "do-nothing"
outflow of advection-diffusion — and hydrogen is carried out at the rate the flow delivers it:

```{code-cell} ipython3
c_open = run(outflow_bc=True)
print(f"peak concentration: {c_open.x.array.max():.1f} times the inlet value")
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt


def plot_field(u, ax, **kwargs):
    coords = u.function_space.tabulate_dof_coordinates()
    return ax.tricontourf(coords[:, 0], coords[:, 1], u.x.array, levels=40, **kwargs)


fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
for ax, u, title in zip(
    axs, [c_closed, c_open], ["no condition on the outlet", "OutflowBC on the outlet"]
):
    cs = plot_field(u, ax)
    fig.colorbar(cs, ax=ax, label="c")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_ylabel("y (m)")
axs[-1].set_xlabel("x (m)")
plt.tight_layout()
plt.show()
```

The closed case builds a thin boundary layer of thickness $D/v$ against the outlet, in which the
concentration climbs far above the inlet value before diffusing sideways into the absorbing walls.
The open case simply washes through.

```{note}
`OutflowBC` is a no-op on a surface where no drift acts on the species, so it is harmless to leave
in place while you experiment with turning the drift on and off.
```

+++

## Surface fluxes carry the drift too

{py:class}`festim.SurfaceFlux` reports the **total** flux $(-D \nabla c + c \mathbf{v}) \cdot
\mathbf{n}$, so a drift term contributes to it.

The clearest way to see that is a case where the diffusive part is exactly zero. Take the same
channel with impermeable walls instead of absorbing ones: hydrogen enters at $c = 1$, nothing
removes it, so the steady state is $c = 1$ everywhere and $\nabla c$ vanishes identically. Anything
`SurfaceFlux` reports is then purely advective.

```{code-cell} ipython3
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L_x, L_y]], [40, 20])

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(mesh)

vol = F.VolumeSubdomain(id=1, material=F.Material(D_0=D, E_D=0))
inlet = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 0.0))
outlet = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], L_x))
walls = F.SurfaceSubdomain(
    id=4, locator=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], L_y)
)
my_model.subdomains = [vol, inlet, outlet, walls]

H = F.Species("H")
my_model.species = [H]

my_model.drift_terms = [
    F.AdvectionTerm(velocity=make_velocity(mesh), subdomain=vol, species=H)
]
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=inlet, value=1.0, species=H),
    F.OutflowBC(subdomain=outlet, species=H),
]

flux_inlet = F.SurfaceFlux(field=H, surface=inlet)
flux_outlet = F.SurfaceFlux(field=H, surface=outlet)
flux_walls = F.SurfaceFlux(field=H, surface=walls)
my_model.exports = [flux_inlet, flux_outlet, flux_walls]

my_model.temperature = 500
my_model.settings = F.Settings(atol=1e-12, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

The solution is uniform, as expected:

```{code-cell} ipython3
c = H.post_processing_solution.x.array
print(f"c ranges from {c.min():.12f} to {c.max():.12f}")
```

`SurfaceFlux` is positive when hydrogen *leaves* the domain, so the inlet reads negative and the
outlet positive. Both are the mean velocity times the channel height, $\tfrac{2}{3} v_{max} L_y$,
and the impermeable walls read zero:

```{code-cell} ipython3
print(f"inlet : {flux_inlet.data[-1]:+.6f}")
print(f"outlet: {flux_outlet.data[-1]:+.6f}")
print(f"walls : {flux_walls.data[-1]:+.2e}")
print(f"expected magnitude (2/3) v_max L_y = {2 / 3 * v_max * L_y:.6f}")

# nothing is created or destroyed, to machine precision
assert abs(flux_inlet.data[-1] + flux_outlet.data[-1] + flux_walls.data[-1]) < 1e-12
# and the flux reported is the advective one, since the diffusive part is zero here
assert np.isclose(flux_outlet.data[-1], 2 / 3 * v_max * L_y, rtol=1e-2)
```

The small shortfall against $\tfrac{2}{3} v_{max} L_y$ is not a FESTIM error: the parabolic velocity
is interpolated into a P1 space, and the piecewise-linear interpolant of a parabola integrates to
slightly less than the parabola. Refining the mesh closes the gap.

+++

## Stabilisation

```{warning}
FESTIM does not stabilise the advection-diffusion form. Where the drift dominates diffusion the
solution oscillates. The quantity to watch is the **cell Péclet number**

$$ \mathrm{Pe}_h = \frac{|\mathbf{v}| \, h}{D} $$

with $h$ the cell size. Keep it around 1 or below; refine the mesh where you cannot.
```

```{code-cell} ipython3
h = L_x / 160
print(f"cell Peclet number in the runs above: {v_max * h / D:.2f}")
```

This is also the constraint that decides how fine a mesh a steep {py:class}`festim.SoretTerm` or
{py:class}`festim.ElectromigrationTerm` profile needs — the drift velocity is what enters
$\mathrm{Pe}_h$, whatever produced it.
