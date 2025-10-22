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

# Concentration #

Initial conditions are an important part of transient simulations for both hydrogen transport and heat-transfer problems. FESTIM defaults to zero values for initial conditions, but can be set using FESTIM's `InitialConditionBase`. 

This tutorial discusses how to set initial conditions for particle concentrations.

Objectives:
* Setting the initial concentration for a single species
* Defining initial concentration conditions for multiple species

+++

## Setting the initial concentration for a single species ##

The `InitialConcentration` class can be used for defining initial conditions for the concentrations, which must defined on a volume subdomain in a FESTIM simulation:

```{code-cell} ipython3
import festim as F

material = F.Material(D_0=1, E_D=0)
vol = F.VolumeSubdomain(id=1, material=material)
H = F.Species("H")

IC = F.InitialConcentration(value=2, species=H, volume=vol)
```

Initial conditions can also be a function of space `x` and temperature `T`, such as:

$$

\text{IC(x, y, T)} = 2x + 3y + 5T

$$

```{code-cell} ipython3
my_function = lambda x,T: 2*x[0] + 3*x[1] + 5*T
IC = F.InitialConcentration(value=my_function, species=H, volume=vol)
```

## Multi-species initial conditions ##

+++

The same class `InitialConcentration` can be used for multiple species, but a separate initial condition is required for each species.

Consider the following 2D example, where we have three species (hydrogen, deuterium, and tritium) with initial and boundary conditions for species' concentrations:

$$ \text{IC (hydrogen)} = 4 $$

$$ \text{IC (deuterium)} = 5 $$

$$ \text{IC (tritium)} = 6 $$

$$ \text{Concentration BC (left, right, top) = 0} $$

First, we can create a 2D unit square mesh and mark the subdomains:

```{code-cell} ipython3
import festim as F
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
import numpy as np

nx, ny = 50, 50
mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)

model = F.HydrogenTransportProblem()
model.mesh = F.Mesh(mesh)

material = F.Material(D_0=1e-2, E_D=0)

top_surface = F.SurfaceSubdomain(id=1, locator = lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator = lambda x: np.isclose(x[1], 0.0))
right_surface = F.SurfaceSubdomain(id=3, locator = lambda x: np.isclose(x[0], 1.0))
left_surface = F.SurfaceSubdomain(id=4, locator = lambda x: np.isclose(x[0], 0.0))

vol = F.VolumeSubdomain(id=1, material=material)

model.subdomains = [top_surface, bottom_surface, left_surface, right_surface, vol]
```

Now, let's define our species and initial conditions. We must create an `InitialConcentration` for each species, then we can pass the initial conditions into the `initial_conditions` attribute:

```{code-cell} ipython3
H = F.Species("H")
D = F.Species("D")
T = F.Species("T")

IC_H = F.InitialConcentration(value=4, species=H, volume=vol)
IC_D = F.InitialConcentration(value=5, species=D, volume=vol)
IC_T = F.InitialConcentration(value=6, species=T, volume=vol)

ICs = [IC_H, IC_D, IC_T]
model.species = [H, D, T]

model.initial_conditions = ICs
```

```{note}
To pass the initial conditions into FESTIM, you must use a list!
```

+++

We also define the zero concentration boundary conditions for each species. Let's run the simulation for 5 seconds with a stepsize of 1 second, and visualize the final concentration profiles (hydrogen on the left, deuterium in the middle, and tritium on the right):

```{code-cell} ipython3
model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_surface, value=0, species=H),
    F.FixedConcentrationBC(subdomain=right_surface, value=0, species=H),
    F.FixedConcentrationBC(subdomain=top_surface, value=0, species=H),
    F.FixedConcentrationBC(subdomain=left_surface, value=0, species=D),
    F.FixedConcentrationBC(subdomain=right_surface, value=0, species=D),
    F.FixedConcentrationBC(subdomain=top_surface, value=0, species=D),
    F.FixedConcentrationBC(subdomain=left_surface, value=0, species=T),
    F.FixedConcentrationBC(subdomain=right_surface, value=0, species=T),
    F.FixedConcentrationBC(subdomain=top_surface, value=0, species=T),
]
model.temperature = 400
model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=50, stepsize=5)

model.initialise()
model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import pyvista
from dolfinx import plot

def make_ugrid(solution):
    topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = solution.x.array.real
    u_grid.set_active_scalars("c")
    return u_grid

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

H_grid = make_ugrid(H.post_processing_solution)
D_grid = make_ugrid(D.post_processing_solution)
T_grid = make_ugrid(T.post_processing_solution)

pl = pyvista.Plotter(shape=(1, 3))
pl.subplot(0, 0)
_ = pl.add_mesh(H_grid,label='h')
pl.view_xy()

pl.subplot(0, 1)
_ = pl.add_mesh(D_grid,label='d')
pl.view_xy()

pl.subplot(0, 2)
_ = pl.add_mesh(T_grid,label='t')
pl.view_xy()

pl.show()
```
