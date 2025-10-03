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

# Advanced functionality #

FESTIM has an internal solver `HeatTransferProblem` for heat transfer problems. This can be helpful for multiphysics coupling or simulations where temperatures cannot be prescribed, but must be solved for. This tutorial shows how to use FESTIM's heat-transfer solvers and coupling it to hydrogen transport simulations.

Objectives:
* Solve steady-state and transient heat-transfer problems in FESTIM
* Couple transient hydrogen transport and heat-transfer simulations

+++

## Temperature field from a steady state heat transfer simulation ##

The governing equation for transient heat transfer is:

$$\rho \ C_p \frac{\partial T}{\partial t} = \nabla \cdot (\lambda \ \nabla T) + \dot{Q} $$

For steady-state problems, the left side of the equation is equal to 0:

$$  \nabla \cdot (\lambda \ \nabla T) + \dot{Q} = 0 $$

Consider the following steady-state heat transfer problem:

```{code-cell} ipython3
import festim as F

heat_transfer_model = F.HeatTransferProblem()
```

We define a thermal conductivity function $ \lambda $ and assign it to our material:  

```{code-cell} ipython3
def thermal_cond_function(T):
    return 3 + 0.1 * T

mat = F.Material(D_0=4.1e-7, E_D=0.39, thermal_conductivity=thermal_cond_function)
```

We can also add heat sources (`HeatSource`), fixed temperature (`FixedTemperatureBC`), and heat flux (`HeatFluxBC`) boundary conditions. We also prescribe a convective heat flux by defining an external temperature $T_{ext}$ and heat transfer coefficient $h$:

```{code-cell} ipython3
import dolfinx
from mpi4py import MPI
import numpy as np

nx = ny = 20
mesh_fenics = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
heat_transfer_model.mesh = F.Mesh(mesh=mesh_fenics)

volume_subdomain = F.VolumeSubdomain(id=1, material=mat)

top_bot = F.SurfaceSubdomain(id=2, locator=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0.0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1.0))

heat_transfer_model.subdomains = [volume_subdomain, top_bot, left, right]

heat_transfer_model.sources = [
    F.HeatSource(value=lambda x: 1 + 0.1 * x[0], volume=volume_subdomain)
]

import ufl

fixed_temperature_left = F.FixedTemperatureBC(
    subdomain=left, value=lambda x: 350 + 20 * ufl.cos(x[0]) * ufl.sin(x[1])
)

def h_coeff(x):
    return 100 * x[0]

def T_ext(x):
    return 300 + 3 * x[1]

convective_heat_transfer = F.HeatFluxBC(
    subdomain=top_bot, value=lambda x, T: h_coeff(x) * (T_ext(x) - T)
)

heat_flux = F.HeatFluxBC(
    subdomain=right, value=lambda x: 10 + 3 * ufl.cos(x[0]) + ufl.sin(x[1])
)

heat_transfer_model.boundary_conditions = [
    fixed_temperature_left,
    convective_heat_transfer,
    heat_flux,
]

heat_transfer_model.settings = F.Settings(
    transient=False,
    atol=1e-09,
    rtol=1e-09,
)

heat_transfer_model.initialise()
heat_transfer_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import pyvista

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

from dolfinx import plot

T = heat_transfer_model.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["T"] = T.x.array.real
u_grid.set_active_scalars("T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")
```

To use this temperature field for a hydrogen transport simulation, we need to define another problem `hydrogen_problem` using `HydrogenTransportProblem`. We simply assign the output from the heat transfer simulation to the temperature attribute of our hydrogen simulation:

```{code-cell} ipython3
hydrogen_problem = F.HydrogenTransportProblem()

hydrogen_problem.mesh = heat_transfer_model.mesh
H = F.Species("H")
hydrogen_problem.species = [H]
hydrogen_problem.temperature = heat_transfer_model.u

hydrogen_problem.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left, value=1e15, species=H),
    F.FixedConcentrationBC(subdomain=right, value=0, species=H),
]

hydrogen_problem.subdomains = heat_transfer_model.subdomains

hydrogen_problem.settings = F.Settings(
    transient=False,
    atol=1e-09,
    rtol=1e-09,
)

hydrogen_problem.initialise()
hydrogen_problem.run()
```

```{code-cell} ipython3
:tags: [hide-input]

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

from dolfinx import plot

c = hydrogen_problem.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")
```

To run a transient heat transfer simulation (with no hydrogen transport coupling), we must add the density $\rho$ and heat capacity $C_p$:

```{code-cell} ipython3
mat = F.Material(D_0=4.1e-7, E_D=0.39, thermal_conductivity=thermal_cond_function, density=20, heat_capacity=50)
volume_subdomain = F.VolumeSubdomain(id=1, material=mat)

heat_transfer_model.subdomains = [volume_subdomain, top_bot, left, right]

heat_transfer_model.settings = F.Settings(
    atol=1e-09,
    rtol=1e-09,
    stepsize=0.1,
    final_time=5
)

heat_transfer_model.initialise()
heat_transfer_model.run() 
```

```{code-cell} ipython3
:tags: [hide-input]

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

T = heat_transfer_model.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["T"] = T.x.array.real
u_grid.set_active_scalars("T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")
```

## Temperature from a transient heat transfer simulation ##

For a coupled heat-transfer and hydrogen transient simulation, we need to use another problem class `CoupledTransientHeatTransferHydrogenTransport` to ensure the solution solves each problem at each step.

First, we must define a `HydrogenTransportProblem` and `HeatTransferProblem` with each model's settings. We also define a common mesh:

```{code-cell} ipython3
import festim as F
import dolfinx
from mpi4py import MPI
import numpy as np
import ufl

nx = ny = 20
mesh_fenics = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
mesh = F.Mesh(mesh=mesh_fenics)


mat = F.Material(D_0=4.1e-4, E_D=0.0, thermal_conductivity=3, density=20, heat_capacity=5)

volume_subdomain = F.VolumeSubdomain(id=1, material=mat)

top_bot = F.SurfaceSubdomain(id=2, locator=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0.0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1.0))
subdomains = [volume_subdomain, top_bot, left, right]

heat_transfer_model = F.HeatTransferProblem()
hydrogen_problem = F.HydrogenTransportProblem()

heat_transfer_model.mesh = mesh   
hydrogen_problem.mesh = mesh

H = F.Species("H")
hydrogen_problem.species = [H]

hydrogen_problem.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_bot, value=1e10, species=H),
    F.FixedConcentrationBC(subdomain=left, value=0, species=H),
]

heat_transfer_model.subdomains = subdomains
hydrogen_problem.subdomains = subdomains

hydrogen_problem.settings = F.Settings(
    transient=True,
    atol=1e-09,
    rtol=1e-09,
    stepsize=1,
    final_time=50
)

fixed_temperature_left = F.FixedTemperatureBC(
    subdomain=left, value=lambda x: 350 + 20 * ufl.cos(x[0]) * ufl.sin(x[1])
)

heat_transfer_model.boundary_conditions = [
    fixed_temperature_left,
]

heat_transfer_model.settings = F.Settings(
    transient=True,
    atol=1e-09,
    rtol=1e-09,
    stepsize=1,
    final_time=50
)
```

Finally, we define and solve a new `problem` using `CoupledTransientHeatTransferHydrogenTransport`:

```{code-cell} ipython3
problem = F.CoupledTransientHeatTransferHydrogenTransport(heat_problem=heat_transfer_model, hydrogen_problem=hydrogen_problem)
problem.initialise()
problem.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import pyvista

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

T = problem.heat_problem.u
c = problem.hydrogen_problem.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["T"] = T.x.array.real
u_grid.set_active_scalars("T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")

    
topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```
