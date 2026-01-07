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

# Using PyVista for interactive visualization #

PyVista has many helpful capabilities for 3D visualization based on VTK code, and is used frequently throughout the FESTIM workshop. Users can use PyVista in Jupyter Notebook to visualize fields and meshes interactively. 

Objectives: 
* Viewing meshes in 3D interactive scenes
* Plotting the solution for continuous problems
* Plotting the solution for discontinuous problems

+++

## Viewing meshes in 3D interactive scenes ##

PyVista is very helpful when plotting 1D/2D/3D meshes, such as this unit square:

```{code-cell} ipython3
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
from dolfinx import plot
import pyvista

nx, ny = 10, 10
mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)

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

## Plotting the solution for a continuous problem ##

FESTIM has some convenience attributes to plot the solution fields in continuous problems.

For example, let's solve a simple 2D diffusion problem on a unit square:

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

We can plot the solution as an unstructured grid using the species' `post_processing_solution` attribute:

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
u_grid = make_ugrid(H.post_processing_solution)
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.view_xy()
u_plotter.add_text("Hydrogen concentration", font_size=12)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

## Plotting the solution for continuous problems ##

Users can also use PyVista to visualize the fields in discontinuous problems.

Let us consider the same __[multi-material problem](https://festim-workshop.readthedocs.io/en/festim2/content/material/material_basics.html#multi-material-example)__ in the Materials chapter, where we use `HydrogenTransportProblemDiscontinuous` to solve a multi-material problem:

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

material_top = F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0)
material_bottom = F.Material(D_0=2, E_D=0, K_S_0=3, E_K_S=0)

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

We must create an unstructured grid for each volume subdomain in our problem using `subdomain_to_post_processing_solution`, which requires a subdomain to be specified:

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

We see each subdomain plotted in the PyVista scene, with a discontinuity at the interface between each subdomain.
