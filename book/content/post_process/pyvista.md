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

```{note}
PyVista is not suitable for transient data, and is used throughout this workshop to display steady-state results or the final timestep.
```

Objectives: 
* Plotting the solution for continuous problems
* Plotting the solution for discontinuous problems
* Inspecting meshes in 3D interactive scenes

+++

## Plotting the solution for a continuous problem ##

FESTIM has some built-in classes to plot the solution fields in continuous problems.

For example, let's solve a simple 2D diffusion problem on a unit square:

```{code-cell} ipython3
:tags: [hide-input]

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

We can visualize the solution as an unstructured grid using the species’ `post_processing_solution` attribute. 

First, we initialize a PyVista `Plotter` object. Using the solution’s function space, we extract the mesh topology, cell types, and geometry with Dolfinx’s `plot.vtk_mesh` function: 

```{code-cell} ipython3
:tags: [hide-input]

import pyvista

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")
```

```{code-cell} ipython3
import pyvista
from dolfinx import plot

plotter = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(H.post_processing_solution.function_space)
```

Next, we create a PyVista `UnstructuredGrid` and attach the solution values as point data using `H.post_processing_solution.x.array.real`. By assigning this array to `grid.point_data["c"]`, we can visualize the species concentration at each mesh point. We then set the active scalars with `grid.set_active_scalars("c")` to use these values for the colormap:

```{code-cell} ipython3
:tags: [hide-output]

grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["c"] = H.post_processing_solution.x.array.real
grid.set_active_scalars("c")
```

Finally, we add the mesh to the plotter with `add_mesh` (optionally specifying a colormap) and display the scene with `plotter.show()`:

```{code-cell} ipython3
plotter.add_mesh(grid, cmap="viridis", show_edges=False)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("concentration.png")
```

```{Tip}
Users can change the colormap by adjusting the `cmap` argument.
```

+++

## Plotting the solution for discontinuous problems ##

Users can also use PyVista to visualize the fields in discontinuous problems.

Let us consider the same __[multi-material problem](https://festim-workshop.readthedocs.io/en/festim2/content/material/material_basics.html#multi-material-example)__ in the Materials chapter, where we use `HydrogenTransportProblemDiscontinuous` to solve a multi-material problem.

```{code-cell} ipython3
:tags: [hide-input]

import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

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

For discontinuous problems, we must create an unstructured grid for each volume subdomain in our problem using `subdomain_to_post_processing_solution`, which requires a subdomain to be specified.

Let us first create an unstructured grid for the top domain:

```{code-cell} ipython3
:tags: [hide-output]

import pyvista
from dolfinx import plot

top_solution = H.subdomain_to_post_processing_solution[top_volume]
topology, cell_types, geometry = plot.vtk_mesh(top_solution.function_space)
top_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
top_grid.point_data["c"] = top_solution.x.array.real
top_grid.set_active_scalars("c")
```

We can repeat the same process for the bottom domain:

```{code-cell} ipython3
:tags: [hide-output]

bottom_solution = H.subdomain_to_post_processing_solution[bottom_volume]
topology, cell_types, geometry = plot.vtk_mesh(bottom_solution.function_space)
bottom_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
bottom_grid.point_data["c"] = bottom_solution.x.array.real
bottom_grid.set_active_scalars("c")
```

Let us now initiate the plotter object, add the solutions to our scene, and visualize the solution:

```{code-cell} ipython3
plotter = pyvista.Plotter()
plotter.add_mesh(top_grid, cmap="magma", show_edges=False)
plotter.add_mesh(bottom_grid, cmap="magma", show_edges=False)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("concentration.png")
```

We see each subdomain plotted in the PyVista scene, with a discontinuity at the interface between each subdomain.

+++

## Inspecting meshes in 3D interactive scenes ##

PyVista is very helpful when plotting 1D/2D/3D meshes, allowing users to examine their mesh, verify mesh/facet tags are correct, and inspecting mesh quality. 

In this example, we want to examine the mesh quality using PyVista. Let us create a 2D unit square mesh: 

```{code-cell} ipython3
:tags: [hide-input]

import pyvista

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")
```

```{code-cell} ipython3
from dolfinx.mesh import create_unit_square
from mpi4py import MPI


nx, ny = 10, 10
mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)
```

Now we need to extract topological dimension of the mesh `mesh.topology.dim` and then create connectivity between a pair of dimensions (creates entities of each dimension and then finds the relation between the entities of different dimensions) using `mesh.topology.create_connectivity`:

```{code-cell} ipython3
from dolfinx import plot
import pyvista

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim, tdim)
```

Once the connectivity is created, we extract the topology, cell types, and geometry similary using `plot.vtk_mesh`, except now we specify which dimension to get mesh data for using `mesh` and `tdim`:

```{code-cell} ipython3
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
```

We create an `UnstructuredGrid` the same way as mentioned above. To view the mesh quality, we can set `set_edges` to `True` when adding the mesh, allowing us to see how refined the mesh is:

```{code-cell} ipython3
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```
