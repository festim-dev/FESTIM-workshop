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

# PyVista #

PyVista has many helpful capabilities for 3D visualization based on VTK code, and is used frequently throughout the FESTIM workshop.

Objectives: 
* Meshes
* Solution (continous)
* Solution (discontinous)

+++

## Using PyVista for interactive visualization ##

PyVista is very helpful when working within a notebook, as you can visualize meshes, results, and more at each step. 

```{code-cell} ipython3
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

nx, ny = 10, 10  # Number of divisions in x and y directions
mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)

from dolfinx import plot
import pyvista

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")


tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```

```{code-cell} ipython3
import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
festim_mesh = F.Mesh(fenics_mesh)

my_model = F.HydrogenTransportProblem()

mat = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain(id=1, material=mat)

top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))

my_model.mesh = festim_mesh
my_model.subdomains = [top_surface, bottom_surface, vol]

H = F.Species("H")
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

def make_ugrid(solution):
    topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = solution.x.array.real
    u_grid.set_active_scalars("c")
    return u_grid

u_plotter = pyvista.Plotter()
u_grid_top = make_ugrid(H.post_processing_solution)
u_plotter.add_mesh(u_grid_top, cmap="magma", show_edges=False)
u_plotter.view_xy()
u_plotter.add_text("Hydrogen concentration in multi-material problem", font_size=12)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

```{code-cell} ipython3
import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
import pyvista
from dolfinx import plot

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
festim_mesh = F.Mesh(fenics_mesh)

my_model = F.HydrogenTransportProblemDiscontinuous()

# Top material (Material A)
material_top = F.Material(
    name="Material_A",
    D_0=1,    # m²/s
    E_D=0,     # eV
    K_S_0=2,    # mol/m³/Pa  (solubility prefactor)
    E_K_S=0,     # eV (activation energy for solubility)
)

# Bottom material (Material B)
material_bottom = F.Material(
    name="Material_B",
    D_0=2,    # m²/s
    E_D=0,     # eV
    K_S_0=3,    # mol/m³/Pa
    E_K_S=0,     # eV
)

top_volume = F.VolumeSubdomain(id=3, material=material_top, locator=lambda x: x[1] >= 0.5)
bottom_volume = F.VolumeSubdomain(id=4, material=material_bottom, locator=lambda x: x[1] <= 0.5)

top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))

my_model.mesh = festim_mesh
my_model.subdomains = [top_surface, bottom_surface, top_volume, bottom_volume]

my_model.interfaces = [F.Interface(5, (bottom_volume, top_volume), penalty_term=1000)]
my_model.surface_to_volume = {
    top_surface: top_volume,
    bottom_surface: bottom_volume,
}

H = F.Species("H")
my_model.species = [H]

for species in my_model.species:
    species.subdomains = [bottom_volume, top_volume]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
def make_ugrid(solution):
    topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = solution.x.array.real
    u_grid.set_active_scalars("c")
    return u_grid

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

u_plotter = pyvista.Plotter()
u_grid_top = make_ugrid(H.subdomain_to_post_processing_solution[top_volume])
u_grid_bottom = make_ugrid(H.subdomain_to_post_processing_solution[bottom_volume])
u_plotter.add_mesh(u_grid_top, cmap="magma", show_edges=False)
u_plotter.add_mesh(u_grid_bottom, cmap="magma", show_edges=False)
u_plotter.view_xy()
u_plotter.add_text("Hydrogen concentration in multi-material problem", font_size=12)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```
