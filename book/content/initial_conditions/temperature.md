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

# Temperature #

This tutorial discusses how to set initial conditions in FESTIM heat-transfer problems using the `InitialConditionBase`.

Objectives:
* Defining temperature initial conditions
* Solving a heat-transfer problem with initial conditions

+++

## Defining temperature initial conditions ##

Similar to the __[concentration](https://festim-workshop.readthedocs.io/en/latest/content/initial_conditions/concentration.html)__, we can define temperature initial conditions on volume subdomains using `InitialTemperature`:

```{code-cell} ipython3
import festim as F

material = F.Material(D_0=1, E_D=0, thermal_conductivity=1, heat_capacity=2,density=1)
vol = F.VolumeSubdomain(id=1, material=material)

IC = F.InitialTemperature(value=400, volume=vol)
```

```{note}
Since initial conditions are only used in transient simulations, `thermal_conductivity`, `heat_capacity`, and `density` must be defined for the material. Learn more about defining thermal properties __[here](https://festim-workshop.readthedocs.io/en/latest/content/material/material_basics.html#defining-thermal-properties)__.
```

+++

Temperature initial conditions can be also be a function of space `x`, such as:

$$ \text{IC(x, y ,z)} = 2\sin(x) + 4 \cos(y) - 2 \exp(-z) $$

```{code-cell} ipython3
import numpy as np

my_temperature = lambda x: 2*np.sin(x[0]) + 4*np.cos(x[1]) - 2*np.exp(-x[2])

IC = F.InitialTemperature(value=my_temperature, volume=vol)
```

## Solving a heat-transfer problem with initial conditions ##

Consider the following 2D heat transfer problem with boundary and initial conditions:

$$ \text{IC} = 300 \text{K} $$

$$ \text{Temperature BC (left)} = 400 \text{K} $$

$$ \text{Temperature BC (right)} = 350 \text{K} $$

Let us first create our mesh and subdomains:

```{code-cell} ipython3
import festim as F
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
import numpy as np

nx, ny = 50, 50
mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)

model = F.HeatTransferProblem()
model.mesh = F.Mesh(mesh)

material = F.Material(D_0=1, E_D=0, thermal_conductivity=1, heat_capacity=2,density=1)

top_surface = F.SurfaceSubdomain(id=1, locator = lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator = lambda x: np.isclose(x[1], 0.0))
right_surface = F.SurfaceSubdomain(id=3, locator = lambda x: np.isclose(x[0], 1.0))
left_surface = F.SurfaceSubdomain(id=4, locator = lambda x: np.isclose(x[0], 0.0))

vol = F.VolumeSubdomain(id=1, material=material)

model.subdomains = [top_surface, bottom_surface, left_surface, right_surface, vol]
```

We can pass our initial conditions using the `initial_conditions` attribute, which requires a list. Let's run the simulation for 5 seconds and see the results:

```{code-cell} ipython3
model.initial_conditions = [
    F.InitialTemperature(value=300, volume=vol)
]

model.boundary_conditions = [
    F.FixedTemperatureBC(subdomain=left_surface, value=400),
    F.FixedTemperatureBC(subdomain=right_surface, value=350),
]

model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=5, stepsize=1)

model.initialise()
model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import pyvista
from dolfinx import plot

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

T = model.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["T"] = T.x.array.real
u_grid.set_active_scalars("T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")
```
