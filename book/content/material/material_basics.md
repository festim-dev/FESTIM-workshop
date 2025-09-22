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

# Basic Material Features #

Materials are key components of hydrogen transport simulations. They hold the properties like diffusivity, solubility and even thermal properties like thermal conductivity or heat capacity. Read more about the `Materials` class and syntax at __[Materials](https://festim.readthedocs.io/en/fenicsx/userguide/subdomains.html)__.


Objectives:
* Learn how to define material properties (thermal, solubility, diffusion)
* Learn how to define materials on different subdomains
* Solve a multi-material diffusion problem

+++

## Defining material properties ##

We can define a material using the diffusion exponential pre-factor $D_0$ and activation energy $E_d$, which is material dependent:

```{code-cell} ipython3
import festim as F

mat = F.Material(D_0=1.11e-6, E_D=0.4)  # m2/s, eV
```

When considering chemical potential conservation at material interfaces, hydrogen solubility can be defined using the solubility coeffeicient prefactor `K_S_0`, solubility activation energy `E_K_S`, and solubility law `solubility_law` (either `"henry"` or `"sievert"`):

```{code-cell} ipython3
mat.K_S_0 = 1.0
mat.E_K_S = 3.0
mat.solubility_law = "sievert"
```

Some material properties are temperature-dependent. Users can define such thermal properties (thermal conductivity, heat capacity, density) as function of temperature in the following way:

```{code-cell} ipython3
import ufl

mat.thermal_conductivity = lambda T: 3*T + 2*ufl.exp(-20*T)
mat.heat_capacity = lambda T: 4*T + 8
mat.density = lambda T: 7*T + 5
```

## Defining materials on different subdomains ##

Volume subdomains are used to assign different materials or define regions with specific physical properties. Each volume subdomain must be associated with a `festim.Material` object. Read more about subdomains __[here](https://festim-workshop.readthedocs.io/en/festim2/content/meshes/mesh_fenics.html)__.

Consider the following volume with two subdomains separated halfway through the mesh:

```{code-cell} ipython3
:tags: [hide-input]

import festim as F
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
festim_mesh = F.Mesh(fenics_mesh)
```

To define one material on each subdomain:

```{code-cell} ipython3
mat1 = F.Material(D_0=1.11e-6, E_D=0.4)  # m2/s, eV

top = F.VolumeSubdomain(id=1, material=mat, locator=lambda x: x[0] >= 0.5)
bottom = F.VolumeSubdomain(id=2, material=mat, locator=lambda x: x[0] < 0.5)
```

Similarly, for two materials:

```{code-cell} ipython3
mat1 = F.Material(D_0=1.11e-6, E_D=0.4)  # m2/s, eV
mat2 = F.Material(D_0=2e-6, E_D=0.3)  # m2/s, eV

top = F.VolumeSubdomain(id=1, material=mat1, locator=lambda x: x[0] >= 0.5)
bottom = F.VolumeSubdomain(id=2, material=mat2, locator=lambda x: x[0] < 0.5)
```

## Solving a multi-material diffusion problem ##
Considering the following 2D example, where hydrogen diffuses through a 2D domain composed of two materials with different diffusion and solubility properties. The top half (Material A) has a higher diffusion coefficient and solubility than the bottom half (Material B). The interface at $𝑦=0.5$ clearly separates the two materials, and the steady-state hydrogen distribution illustrates how material properties impact transport.

+++

First, create the mesh:

```{code-cell} ipython3
import festim as F
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
import pyvista
from dolfinx import plot

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 40, 40)  # finer resolution
festim_mesh = F.Mesh(fenics_mesh)
```

Then, we define materials:

```{code-cell} ipython3
# Top material (Material A) - "Cu-like"
material_A = F.Material(
    name="Material_A",
    D_0=1e-6,    # m²/s
    E_D=0.3,     # eV
    K_S_0=5e-3,    # mol/m³/Pa  (solubility prefactor)
    E_K_S=0.1,     # eV (activation energy for solubility)
    thermal_conductivity=400,  # W/m/K
    heat_capacity=385, # J/kg/K
    density=8960       # kg/m³
)

# Bottom material (Material B) - "W-like"
material_B = F.Material(
    name="Material_B",
    D_0=2e-7,    # m²/s
    E_D=1.2,     # eV
    K_S_0=2e-3,    # mol/m³/Pa
    E_K_S=0.2,     # eV
    thermal_conductivity=180,  # W/m/K
    heat_capacity=140, # J/kg/K
    density=19300      # kg/m³
)
```

Now subdomains:

```{code-cell} ipython3
top_volume = F.VolumeSubdomain(id=1, material=material_A, locator=lambda x: x[1] >= 0.5)
bottom_volume = F.VolumeSubdomain(id=2, material=material_B, locator=lambda x: x[1] <= 0.5)

top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))
```

Finally, we define the problem, boundary conditions, and then solve:

```{code-cell} ipython3

my_model = F.HydrogenTransportProblem()
my_model.mesh = festim_mesh
my_model.subdomains = [top_surface, bottom_surface, top_volume, bottom_volume]

# Species
H = F.Species("H")
my_model.species = [H]

# Temperature (constant across whole domain for now)
my_model.temperature = 600  # K

# Boundary conditions
my_model.boundary_conditions = [
    F.DirichletBC(subdomain=top_surface, value=1.0, species=H),
    F.DirichletBC(subdomain=bottom_surface, value=0.0, species=H),
]

# Solver settings
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

# Run simulation
my_model.initialise()
my_model.run()
```

Visualizing the results:

```{code-cell} ipython3
hydrogen_concentration = H.solution

topology, cell_types, geometry = plot.vtk_mesh(hydrogen_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = hydrogen_concentration.x.array.real
u_grid.set_active_scalars("c")

u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="magma", show_edges=False)
u_plotter.view_xy()
u_plotter.add_text("Hydrogen concentration in multi-material problem", font_size=12)
interface_line = pyvista.Line(pointa=(0, 0.5, 0), pointb=(1, 0.5, 0))
u_plotter.add_mesh(interface_line, color="black", line_width=2, label="Material Interface")

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```
