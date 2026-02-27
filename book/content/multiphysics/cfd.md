---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

# Coupling to computational fluid dynamics #

This tutorial discusses how to couple FESTIM to computational fluid dynamics (CFD) solvers, which is relevant in modeling advective-assisted and turbulent diffusion.


```{admonition} Objectives
:class: objectives
* Understanding the importance of fluid flow in hydrogen diffusion
* Utilizing external CFD solvers to generate velocity fields
* Reading fields into FESTIM for advective-assisted diffusion
```
+++

## Importance of fluid flow in hydrogen diffusion

Let's review the governing equation for FESTIM, neglecting reactions and source terms:

$$
\frac{\partial c_m}{\partial t} = \nabla \cdot (D \nabla c_m) + \mathbf{u} \cdot \nabla c_m ,
$$

where $\mathbf{u}$ is the velocity field in units of $\mathrm{ms^{-1}}$. This velocity term, which represents the species' advection, requires solving for the $\textbf{u}$ field. This can be done using custom (DOLFINx) or external (OpenFOAM) CFD solvers. 

Additionally, turbulence is a diffusion-enhancing property. This can be incorporated in FESTIM by adding a turbulent diffusion term $D_t$:

$$ D_t = \frac{\nu_t}{Sc} $$

where $\nu_t$ is the turbulent viscosity and $Sc$ is the Schmidt number (measures the relative rates of mass and momentum diffusion). This turbulent diffusion term is simply added to the Fickian diffusivity in FESTIM.

+++

### When do I need advection in my diffusion model? 

Advection can add complexity to your FESTIM model, and sometimes may not be needed. Users can choose whether or not they need advection by referring to the Péclet number $Pe$:

$$ Pe = \frac{\text{advection transport rate}}{\text{diffusive transport rate}} = \frac{\mathrm{Lu}}{D} = \mathrm{Re_L}\mathrm{Sc} $$

where $L$ is the characteristic length, $u$ the local flow velocity, $D$ the mass diffusion coefficient, $Re_L$ the Reynolds number, $Sc$ the Schmidt number. If $Pe << 1$, diffusion dominates and advection is not needed.

+++

## Solving a lid-driven cavity problem

This section discusses how to setup and solve a coupled problem between OpenFOAM and FESTIM, as outlined in the FESTIM 2 review paper. Specifically, we'll solve a lid-driven cavity problem by calculating the velocity field in OpenFOAM and exporting it to FESTIM to solve our diffusion model.

```{seealso}
Check out the [FESTIM 2 review paper](https://arxiv.org/abs/2509.24760) to learn more.
```

+++

Our rectangular geometry will have the following velocity boundary conditions:

$$
\mathbf{u}(x=0, y) = (0, 0), \quad
\mathbf{u}(x=L, y) = (0, 0), \\
\mathbf{u}(x, y=0) = (0, 0), \quad
\mathbf{u}(x, y=L) = (U, 0)
$$

+++

Let us first import our OpenFOAM data (which has been stored on the [FESTIM 2 review paper repo](https://github.com/festim-dev/FESTIM-v2-review/tree/main/coupling/coupling_cfd/data).)

```{seealso}
The OpenFOAM case for this problem can be found [here](https://github.com/festim-dev/FESTIM-v2-review/blob/main/coupling/coupling_cfd/ldc_coupling.py). This problem uses the *icoFOAM* solver, take a look at the [OpenFOAM documentation](https://www.openfoam.com/documentation/tutorial-guide/2-incompressible-flow/2.1-lid-driven-cavity-flow#x6-60002.1) for more information about running an OpenFOAM case.
```

```{code-cell} ipython3
from foam2dolfinx import OpenFOAMReader
from mpi4py import MPI
from pathlib import Path
import zipfile
from dolfinx.io import VTXWriter

export_field = True
zip_path = Path("cfd_data/cavity.zip")
extract_path = Path("cfd_data/cavity")

# Extract if needed
if not extract_path.exists():
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

# Ensure .foam file exists
foam_file = extract_path / "cavity" / "cavity.foam"
foam_file.touch(exist_ok=True)

# Read OpenFOAM case
ldc_reader = OpenFOAMReader(
    filename=str(foam_file),
    cell_type=12
)

velocity_field = ldc_reader.create_dolfinx_function(t=2.5, name="U")
```

We can visualize the velocity field using PyVista:

```{code-cell} ipython3
:tags: [hide-input]

from dolfinx import plot
import pyvista
import numpy as np

pyvista.set_jupyter_backend("html")

V = velocity_field.function_space
topology, cell_types, geometry = plot.vtk_mesh(V)

grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

num_dofs = V.dofmap.index_map.size_local
value_size = V.dofmap.index_map_bs  # number of components (e.g. 2 or 3)

u_array = velocity_field.x.array.real.reshape(num_dofs, value_size)

grid.point_data["U"] = u_array
grid.set_active_vectors("U")

grid["U_mag"] = np.linalg.norm(u_array, axis=1)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, scalars="U_mag", cmap="coolwarm")
glyphs = grid.glyph(orient="U", factor=0.02)
plotter.add_mesh(glyphs, cmap="coolwarm")
plotter.view_xy()
plotter.show()
```

````{tip}
To save the OpenFOAM velocity field, you can export it to `.bp` and view the results in ParaView (as shown in the [post-processing section](../post_process/paraview.md)).
```
writer = VTXWriter(
    MPI.COMM_WORLD,
    "results/velocity_field.bp",
    velocity_field,
    "BP5"
)
writer.write(t=0)
```
````

+++

Now, let's initiate our hydrogen transport problem. We have insulated boundary conditions ($c=0$ on all boundaries):

```{code-cell} ipython3
import festim as F
from mpi4py import MPI
from dolfinx.mesh import create_rectangle
import numpy as np

my_model = F.HydrogenTransportProblem()
fenics_mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [0.1, 0.1]], [50, 50])
my_model.mesh = F.Mesh(fenics_mesh)

# define species
H = F.Species("H", mobile=True)
my_model.species = [H]

# define subdomains
my_mat = F.Material(D_0=5e-04, E_D=0)

fluid = F.VolumeSubdomain(id=1, material=my_mat)
left = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 0.0))
right = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0.1))
bottom = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[1], 0.0))
top = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[1], 0.1))
my_model.subdomains = [fluid, left, right, bottom, top]

# define boundary conditions
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left, value=0, species=H),
    F.FixedConcentrationBC(subdomain=right, value=0, species=H),
    F.FixedConcentrationBC(subdomain=bottom, value=0, species=H),
    F.FixedConcentrationBC(subdomain=top, value=0, species=H),
]

# define temperature
my_model.temperature = 500
```

In this example, we add a particle source using `ParticleSource':

```{code-cell} ipython3
my_model.sources = [
    F.ParticleSource(volume=fluid, species=H, value=10),
]
```

To add the velocity field to our FESTIM model, we need to create an `AdvectionTerm`, which requires the velocity field (in our case the exported field from OpenFOAM), which species the velocity field is acting on, and which volume subdomain is this velocity field on:

```{code-cell} ipython3
advection = F.AdvectionTerm(velocity=lambda t: velocity_field(t), species=H, subdomain=fluid)
```

```{note}
We use a `lambda` function to utilize transient velocity fields, although our example here is steady-state and therefore only uses data at $t=2.5$ seconds.
```

+++

We add this to our problem's `advection_terms` atribute:

```{code-cell} ipython3
my_model.advection_terms = [advection]
```

Let us solve and visualize the results:

```{code-cell} ipython3
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

pyvista.set_jupyter_backend("html")

c = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()

u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

Compare this to the results without advection:

```{code-cell} ipython3
my_model.advection_terms = []
my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

c = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()

u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```
