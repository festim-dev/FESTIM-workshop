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
* Learn how to define material properties (thermal, solubility, diffusivity)
* Learn how to define materials on different subdomains
* Solve a multi-material diffusion problem

+++

## Defining material properties ##

We can define a material using the diffusion exponential pre-factor $D_0$ and activation energy $E_D$. By default, FESTIM assumes they follow an Arrhenius law, which is of the following form:

$$
    D = D_0 \exp{(-E_D/k_B T)}
$$

where $k_B$ is the Boltzmann constant in eV/K and $T$ is the temperature in K. To define a material using these two properties:

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

## Defining thermal properties ## 

Users can define thermal properties, such as thermal conductivity, heat capacity, and density for their materials. The simplest way to define them:

```{code-cell} ipython3
mat.thermal_conductivity = 10.0  # W/m/K
mat.heat_capacity = 500.0  # J/kg/K
mat.density = 8000.0  # kg/m3
```

Users can also define these as a function of temperature:

```{code-cell} ipython3
import ufl

mat.thermal_conductivity = lambda T: 3*T + 2*ufl.exp(-20*T)
mat.heat_capacity = lambda T: 4*T + 8
mat.density = lambda T: 7*T + 5
```

## Defining materials on different subdomains ##

Volume subdomains are used to assign different materials or define regions with specific physical properties. Each volume subdomain must be associated with a `festim.Material` object. Read more about subdomains __[here](https://festim-workshop.readthedocs.io/en/festim2/content/meshes/mesh_fenics.html#defining-subdomains)__.

Consider the following volume with two subdomains separated halfway through the mesh:

+++

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

## Multi-material example ##

Considering the following 2D example, where hydrogen diffuses through a 2D domain composed of two materials with different diffusion and solubility properties. The top half (Material A) has a lower diffusion coefficient and solubility than the bottom half (Material B). The interface at $𝑦=0.5$ clearly separates the two materials, and the steady-state hydrogen distribution illustrates how material properties impact transport.

+++

First, we create the mesh for our discontinuous (materials have different solubility properties) problem. Note that we use `HydrogenTransportProblemDiscontinuous` to define our multi-material problem:

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
```

<div class="alert alert-block alert-info">
Be sure to use an even number of cells in each direction when creating the mesh in a discontinuous problem.
</div>

+++

Next, we define solubility properties. If the solubilties were the same, we'd expect to see a smooth "continous" concentration profile. For this problem, we have different solubilities:

```{code-cell} ipython3
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
```

Now we must assemble the subdomains for our volumes, surfaces, interfaces, and species. A penalty term is also used when defining the interface to ensure a well-behaved solution at the interface between both materials:

```{code-cell} ipython3
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
```

Finally, we define the temperature, boundary conditions, and then solve:

```{code-cell} ipython3
my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

To visualize the results of the multi-material problem, we need to look at each subdomain separately. We can use ` H.subdomain_to_post_processing_solution ` to define a plotting function `make_ugrid` that visualizes each domain:

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

<div class="alert alert-block alert-success">
Sucess! We see diffusion from the top surface downwards, with a discontinuity at the interface! 
</div>
