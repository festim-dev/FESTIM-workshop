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

# Enclosures on 1D and 2D geometries

An enclosure is not tied to a single wall or a single mesh. Its `surfaces` argument maps **any** number of
contact surfaces to their area, and those surfaces can even belong to physically disconnected pieces of the
domain. This page shows two cases:

* a **discontiguous 1D mesh** — two separate slabs sharing one enclosure;
* a **2D geometry** — a square solid with a quarter-circle void acting as the enclosure.

+++

## Two slabs sharing an enclosure (discontiguous 1D mesh)

Since FESTIM 2.2, `festim.Mesh1D` accepts a **list** of vertex arrays, producing a mesh with
disconnected blocks:

```python
F.Mesh1D([np.linspace(0, 0.5, 30), np.linspace(1.0, 1.5, 30)])
```

Here two slabs are separated by a physical gap, and the gap is filled by an enclosure. There is no
`Interface` between the slabs and no cell spans the gap, so the **only** path from one slab to the other is
through the gas. Hydrogen starts in the left slab, desorbs into the enclosure through a surface reaction,
and is absorbed by the right slab.

```{code-cell} ipython3
import numpy as np
import festim as F

area = 0.25
V_enc, T = 1e-3, 500.0
c_0 = 1e18
k_d0, k_r0 = 1e17, 1e-21
dt, final_time = 5.0, 200.0

my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh1D([np.linspace(0, 0.5, 30), np.linspace(1.0, 1.5, 30)])

material = F.Material(D_0=1e-6, E_D=0.0, K_S_0=1, E_K_S=0)
vol_left = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=material)
vol_right = F.VolumeSubdomain1D(id=2, borders=[1.0, 1.5], material=material)
# the two surfaces facing the gap
inner_left = F.SurfaceSubdomain1D(id=1, x=0.5)
inner_right = F.SurfaceSubdomain1D(id=2, x=1.0)
my_model.subdomains = [vol_left, vol_right, inner_left, inner_right]
my_model.interfaces = []

H = F.Species("H", subdomains=[vol_left, vol_right])
my_model.species = [H]
my_model.temperature = T
```

The enclosure is in contact with **both** inner surfaces — its `surfaces` dict simply lists both:

```{code-cell} ipython3
H2 = F.GasSpecies(name="H2", initial_pressure=0.0)
my_model.enclosures = [
    F.Enclosure(
        volume=V_enc,
        species=[H2],
        temperature=T,
        surfaces={inner_left: area, inner_right: area},
    )
]
my_model.boundary_conditions = [
    F.SurfaceReactionBC(
        reactant=[H, H], gas_pressure=H2,
        k_r0=k_r0, E_kr=0.0, k_d0=k_d0, E_kd=0.0,
        subdomain=surface,
    )
    for surface in (inner_left, inner_right)
]

my_model.initial_conditions = [
    F.InitialConcentration(value=c_0, species=H, volume=vol_left),
    F.InitialConcentration(value=0.0, species=H, volume=vol_right),
]

left_inventory = F.TotalVolume(field=H, volume=vol_left)
right_inventory = F.TotalVolume(field=H, volume=vol_right)
pressure = F.GasPressure(field=H2)
my_model.exports = [left_inventory, right_inventory, pressure]

my_model.settings = F.Settings(atol=1e-8, rtol=1e-10, transient=True,
                               final_time=final_time, stepsize=F.Stepsize(dt))
my_model.show_progress_bar = False
my_model.initialise()
my_model.run()
```

Hydrogen leaves the left slab, transits through the gas (a rise then fall of the enclosure pressure) and
accumulates in the right slab, with the total atom count conserved throughout:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

t = np.array(pressure.t)
in_left = area * np.array(left_inventory.data)
in_right = area * np.array(right_inventory.data)
in_gas = 2.0 * np.array(pressure.data) * V_enc / (F.k_B_SI * T)
total = in_left + in_right + in_gas

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(t, np.array(pressure.data))
ax1.set(xlabel="time (s)", ylabel="H2 pressure (Pa)", title="Enclosure pressure")
ax1.grid(alpha=0.3)

ax2.plot(t, in_left, label="left slab")
ax2.plot(t, in_right, label="right slab")
ax2.plot(t, in_gas, label="enclosure")
ax2.plot(t, total, "k--", label="total (conserved)")
ax2.set(xlabel="time (s)", ylabel="H atoms", title="Inventory")
ax2.legend()
ax2.grid(alpha=0.3)
plt.show()

drift = abs(total[-1] - total[0]) / total[0]
print(f"inventory drift: {drift:.2e} (relative)")
assert drift < 1e-6
```

## A quarter-circle enclosure in a square domain (2D)

In 2D a contact surface is a line, so the enclosure area is the **out-of-plane depth** of the model. Here we
build a square solid (1 mm side) with a quarter-circle void cut out of one corner; the curved boundary of
the void is the enclosure interface. The solid starts loaded with hydrogen, its outer edges are left as
no-flux boundaries (the default), so all the hydrogen desorbs into the void and fills the enclosure.

We build the geometry with the GMSH Python API (see also [](../meshes/mesh_gmsh)) and convert it to a
DOLFINx mesh:

```{code-cell} ipython3
from mpi4py import MPI
import gmsh
from dolfinx.io import gmsh as gmshio

L, R = 1e-3, 5e-4  # square side and void radius (m)

gmsh.initialize()
gmsh.model.add("quarter_circle")
square = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
void = gmsh.model.occ.addDisk(0, 0, 0, R, R)
gmsh.model.occ.cut([(2, square)], [(2, void)])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [gmsh.model.getEntities(dim=2)[0][1]], 1)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), L / 25)
gmsh.model.mesh.generate(2)
mesh = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2).mesh
gmsh.finalize()
```

The arc is identified with a `locator` function — it is where $x^2 + y^2 = R^2$. We don't tag the outer
edges: with no boundary condition there, they are no-flux by default.

```{code-cell} ipython3
c_0, D_0, T = 1e21, 1e-7, 600.0  # initial loading, diffusivity, temperature
VOLUME, DEPTH = 1e-8, 1.0         # enclosure volume (m3) and out-of-plane depth (m)

my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh(mesh=mesh)

material = F.Material(name="tungsten", D_0=D_0, E_D=0.2)
vol = F.VolumeSubdomain(id=1, material=material)
arc = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(np.hypot(x[0], x[1]), R, atol=L / 20))
my_model.subdomains = [vol, arc]

H = F.Species("H", subdomains=[vol])
my_model.species = [H]
my_model.temperature = T

H2 = F.GasSpecies(name="H2", initial_pressure=0.0)
my_model.enclosures = [
    F.Enclosure(volume=VOLUME, species=[H2], temperature=T, surfaces={arc: DEPTH})
]
my_model.boundary_conditions = [
    F.SurfaceReactionBC(reactant=[H, H], gas_pressure=H2,
                        k_r0=1e-26, E_kr=0.0, k_d0=1e14, E_kd=0.0, subdomain=arc),
]
my_model.initial_conditions = [F.InitialConcentration(value=c_0, species=H, volume=vol)]

pressure = F.GasPressure(field=H2)
my_model.exports = [pressure]
my_model.settings = F.Settings(atol=1e10, rtol=1e-10, transient=True,
                               final_time=200.0, stepsize=F.Stepsize(5.0))
my_model.show_progress_bar = False
my_model.initialise()
my_model.run()
print(f"enclosure pressure: {H2.value:.2f} Pa")
```

The concentration field is visualised in one line with `festim.plot`. Because the problem is solved
on a submesh, we pass the volume subdomain — notice the depletion near the arc, where hydrogen leaves the
solid for the gas:

```{code-cell} ipython3
:tags: [hide-input]

import pyvista

pyvista.set_jupyter_backend("html")
F.plot(H, subdomain=vol, show_edges=False)
```
