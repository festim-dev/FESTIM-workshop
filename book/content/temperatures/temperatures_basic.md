---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

# Basic functionality #

Definition of a temperature field or problem is essential for hydrogen transport and FESTIM as a whole. Users can define it as a fixed value or as a function of space and time. Read more about temperature in FESTIM _[here](https://festim.readthedocs.io/en/fenicsx/userguide/temperature.html)_. FESTIM's unit for temperature is Kelvin.

This tutorial will go over the basics of defining the temperature in a FESTIM simulation.

Objectives:
* Learn how to define homogeneous temperature
* Learn how to define space and time dependent temperature functions
* Learn how to define temperature as a `dolfinx.fem.Function`

+++

## Defining temperatures in FESTIM ##

The simplest way to define temperature is as a homogeneous value:

```{code-cell} ipython3
import festim as F

my_model = F.HydrogenTransportProblem()
my_model.temperature = 300  # K 
```

To define a space-dependent temperature function we can use ` lambda ` functions:

\begin{equation}
    T(x) = T_0 \exp(-x)
\end{equation}

```{code-cell} ipython3
import festim as F
import ufl

T0 = 300
my_model = F.HydrogenTransportProblem()
my_model.temperature = lambda x: T0 * ufl.exp(-x[0])  
```

For a time-dependent function such as:

$$

T(t) = T_0 \sin(t)

$$

```{code-cell} ipython3
my_model.temperature = lambda t: T0 * ufl.sin(t)
```

For a function in $ x, y, z $ space:

$$

T(x,y,z) = T_0 [\cos(x)\sin(y) - 2z]

$$

```{code-cell} ipython3
my_model.temperature = lambda x: T0 * (ufl.cos(x[0])*ufl.sin(x[1]) - 2*x[2])
```

```{note}
When defining custom functions for values, only the arguments x, t and T can be defined. Spatial coordinates can be referred to by their indices, such as `x[0]`, `x[1]`, and `x[2]`, regardless of the coordinate system used. Time dependence must use t, and T for temperature dependence.
```

+++

We can assign a unit cube mesh and run the simulation to see how a spatially-dependent temperature field affects the transport:

```{code-cell} ipython3
from dolfinx.mesh import create_unit_cube
from mpi4py import MPI
import ufl
import festim as F
import numpy as np

mesh = F.Mesh(create_unit_cube(MPI.COMM_WORLD, 10, 10, 10))
my_model.mesh = mesh

mat = F.Material(D_0=1, E_D=0)

volume = F.VolumeSubdomain(id=1, material=mat)
top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[2], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[2], 0.0))
my_model.subdomains = [top_surface, bottom_surface, volume]

H = F.Species("H")
my_model.species = [H]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

from dolfinx import plot
import pyvista
from basix.ufl import element
import dolfinx

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

el = element("Lagrange", mesh.mesh.topology.cell_name(), 3)
V = dolfinx.fem.functionspace(mesh.mesh, el)
temperature = dolfinx.fem.Function(V)

coords = ufl.SpatialCoordinate(temperature.function_space.mesh)
x = coords[0]
y = coords[1]
z = coords[2]

interpolation = temperature.function_space.element.interpolation_points
expr = dolfinx.fem.Expression(T0 * ufl.cos(x)*ufl.sin(y) - 2*z, interpolation)                    
temperature.interpolate(expr)

u_plotter = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid.point_data["T"] = temperature.x.array.real
function_grid.set_active_scalars("T")
u_plotter.add_mesh(function_grid, cmap="inferno", show_edges=False, opacity=1)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

c = H.solution

topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()

u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

## Defining temperature as a FEniCS function ##

Users can pass a dolfinx `Function` to the temperature attribute, which can be helpful when using results from other programs (ie. __[converting an OpenFOAM output into a passable dolfinx function](https://github.com/festim-dev/foam2dolfinx/tree/main)__).

For example, we want to have a temperature function defined as:

$$

T(x,y) = 300 \exp{-[(x-0.5)^2 + (y-0.5)^2]}

$$

To define the temperature as a function or expression, we can use `ufl`:

```{code-cell} ipython3
import dolfinx
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
import festim as F
from basix.ufl import element

mesh_fenics = create_unit_square(MPI.COMM_WORLD, 20, 20)
my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(mesh_fenics)

el = element("Lagrange", mesh_fenics.topology.cell_name(), 2)
V = dolfinx.fem.functionspace(my_model.mesh.mesh, el)

temperature = dolfinx.fem.Function(V)

coords = ufl.SpatialCoordinate(temperature.function_space.mesh)
x = coords[0]
y = coords[1]

interpolation = temperature.function_space.element.interpolation_points
expr = dolfinx.fem.Expression(300*ufl.exp(-((x-0.5)**2 + (y-0.5)**2)), interpolation)
                                
temperature.interpolate(expr)
```

```{code-cell} ipython3
:tags: [hide-input]

from dolfinx import plot
import pyvista

topology, cell_types, geometry = plot.vtk_mesh(V)
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid.point_data["T"] = temperature.x.array.real
function_grid.set_active_scalars("T")

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

plotter = pyvista.Plotter()
plotter.add_mesh(function_grid, cmap="inferno", show_edges=False, opacity=1)
plotter.view_xy()
plotter.add_text("Temperature", font_size=12)

if not pyvista.OFF_SCREEN:
    plotter.show()
```

To set this temperature for our FESTIM simulation, we simply need to pass it to our model's temperature attribute:

```{code-cell} ipython3
my_model.temperature = temperature
```
