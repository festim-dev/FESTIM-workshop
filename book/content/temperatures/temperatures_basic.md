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
* Learn how to define fixed temperature values
* Learn how to define space and time dependent temperature functions
* Learn how to define temperature as a Dolfinx expression

+++

## Defining temperatures in FESTIM ##

The simplest way to define temperature is as a fixed value:

```{code-cell} ipython3
import festim as F

my_model = F.HydrogenTransportProblem()
my_model.temperature = 300  # K 
```

To define a temperature function we can use ` lambda ` functions, such as this space-dependent function:

$$ 

T(x) = T_0 e^{-x}

$$

```{code-cell} ipython3
import festim as F
import ufl

T0 = 300
my_model = F.HydrogenTransportProblem()
my_model.temperature = lambda x: T0 * ufl.exp(-x)  # K
```

For a time dependent function such as:

$$

T(t) = T_0 sin(t)

$$

```{code-cell} ipython3
my_model.temperature = lambda t: T0 * ufl.sin(t)
```

For a function in $ x, y, z, t $ space:

$$

T(x,y,z,t) = T_0 cos(x)sin(t) + 5y - 2z

$$

```{code-cell} ipython3
my_model.temperature = lambda x,t: T0 * ufl.cos(x[0])*ufl.sin(t) + 5*x[1] - 2*x[2]
```

<div class="alert alert-block alert-info">
When defining custom functions for values, only the arguments x, t and T can be defined. Spatial coordinates can be referred to by their indices, such as x[0], x[1], and x[2], regardless of the coordinate system used. Time dependence must use t, and T for temperature dependence.
</div>

+++

## Defining temperature as a FEniCS expression or function ##

Users can define temperature as a Dolfinx `Function`, which can be helpful when implementing multiphysics simulations (such as in OpenFOAM or OpenMC). For example, we want to have temperature function defined as:

$$

T(x,y) = 300 e^{-((x-0.5)^2 + (y-0.5)^2)}

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
x = ufl.SpatialCoordinate(temperature.function_space.mesh)[0]
y = ufl.SpatialCoordinate(temperature.function_space.mesh)[1]
interpolation = temperature.function_space.element.interpolation_points()
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
