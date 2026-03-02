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

# Coupling to neutronics

This tutorial discusses how to couple FESTIM to neutronics solvers such as OpenMC.

```{admonition} Objectives
:class: objectives
* Understanding the importance of neutronics in hydrogen transport
* Utilizing external neutronics solvers to generate tritium source terms
* Reading OpenMC results into FESTIM for accurate particle sources
```

+++

## Where does neutronics couple into hydrogen transport? ##

Let us take a look at the governing equation for FESTIM, neglecting reactions and advection:

$$ \frac{\partial c_i}{\partial t} = \nabla\cdot (D_i \nabla c_i) + S_i(\mathbf{x}, t, T) $$

Neutronics solvers can provide a value for the $S_i$, which represents a volumetric source of mobile hydrogen isotope that can depend on space, time and temperature. In particular, solvers such as OpenMC can provide an accurate source of of tritium generation.

### Converting data from OpenMC to FESTIM

OpenMC can tally spatial distributions of reaction rates and export the results in VTK format. Users can then use the `openmc2dolfinx` package to convert them into DOLFINx functions.

```{seealso}
Check out the [OpenMC documentation](https://docs.openmc.org/en/stable/) and [openmc2dolfinx repo](https://github.com/festim-dev/foam2dolfinx) to learn more.
```

+++

## Solving a hydrogen transport problem in an neutron-irradiated lithium cube 

This section discusses how to setup and solve a coupled problem between OpenMC and FESTIM, as outlined in the FESTIM 2 review paper. 

Specifically, we'll calculate the tritium generation in a neutron-irradiated lithium cube using OpenMC. Then, we will export the source term to FESTIM to solve our diffusion model.

```{seealso}
Check out the [FESTIM 2 review paper](https://arxiv.org/abs/2509.24760) to learn more.
```

+++

### Getting tritium source from OpenMC

First, run the OpenMC model:

```
import openmc
import openmc_data_downloader as odd

dim = 60
lithium = openmc.Material(name="lithium")
lithium.set_density("g/cc", 0.534)
lithium.add_element("Li", 1.0)

mats = openmc.Materials([lithium])

odd.download_cross_section_data(
    mats,
    libraries=["FENDL-3.1d"],
    set_OPENMC_CROSS_SECTIONS=True,
    particles=["neutron"],
)

# Geometry
cube_surface = openmc.model.RectangularParallelepiped(
    -dim, dim, -dim, dim, -dim, dim
)
region = -cube_surface
cell = openmc.Cell(region=region, fill=lithium)

vacuum_surf = openmc.Sphere(r=dim * 2, boundary_type="vacuum")
vacuum_region = +cube_surface & -vacuum_surf
vacuum = openmc.Cell(region=vacuum_region, fill=None)

geometry = openmc.Geometry([cell, vacuum])

# Tallies
tally = openmc.Tally(name="tritium_production")
tally.scores = ["(n,Xt)"]
mesh = openmc.RegularMesh()
mesh.dimension = [30, 30, 15]
mesh.lower_left = [-dim, -dim, -dim]
mesh.upper_right = [dim, dim, dim]
tally.filters = [openmc.MeshFilter(mesh)]

tallies = openmc.Tallies([tally])

# Settings
source = openmc.IndependentSource()
source_pos_z = dim + 10
source.space = openmc.stats.Point((0, 0, source_pos_z))
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([14.1e6], [1.0])

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = source
settings.batches = 10
settings.particles = 10000

my_model = openmc.Model(
    geometry=geometry, settings=settings, materials=mats, tallies=tallies
)

my_model.run(apply_tally_results=True)

mesh.write_data_to_vtk(
    "tritium_production_mesh.vtk", {"tritium_production": tally.mean}
)
```

This should output a file named `tritium_production_mesh.vtk` which can then be read into FESTIM.

+++

````{tip}
We provide the code to run the OpenMC model, but don't run it this tutorial. To run this code, create a new environment and install OpenMC with the following lines of code:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --name openmc-env openmc
conda activate openmc-env
```
It is also necessary to install `vtk` and `openmc_data_downloader`, which can be done with:

```
conda install -c conda-forge vtk
pip install openmc_data_downloader
```
````

+++

Let's take a look at the OpenMC tritium source output:

```{code-cell} ipython3
:tags: [hide-input]

import pyvista as pv

pv.set_jupyter_backend("html")

grid = pv.read("openmc_data/tritium_production_mesh.vtk")
scalar_name = "tritium_production"  
grid.set_active_scalars(scalar_name)

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars=scalar_name, cmap="hot", show_edges=False)
plotter.view_xy()
plotter.show()
```

### Read data from OpenMC output

Now that we have our tritium source data, we can use `openmc2dolfinx` to create a structured grid reader that outputs a `dolfinx.fem.Function` to be used in FESTIM:

```{code-cell} ipython3
from openmc2dolfinx import StructuredGridReader

reader = StructuredGridReader("openmc_data/tritium_production_mesh.vtk")
reader.create_dolfinx_mesh()
```

### Setting up our FESTIM model

Let us set up our FESTIM model now. We will create a refined box mesh and use a dummy material. We will consider insulated boundary conditions ($c=0$ on all boundaries):

```{code-cell} ipython3
import festim as F
import dolfinx 
from mpi4py import MPI

dim = 60
my_model = F.HydrogenTransportProblem()

tritium = F.Species("T")

refined_mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    points=[(-dim, -dim, -dim), (dim, dim, dim)],
    n=(30, 30, 70),
    cell_type=dolfinx.mesh.CellType.tetrahedron,
)

my_model.mesh = F.Mesh(refined_mesh)
boundary = F.SurfaceSubdomain(id=1)

my_mat = F.Material(D_0=1, E_D=0)
volume = F.VolumeSubdomain(id=1, material=my_mat)
my_model.subdomains = [boundary, volume]

my_model.species = [tritium]
my_model.boundary_conditions = [F.FixedConcentrationBC(subdomain=boundary, value=0.0, species=tritium)]
my_model.temperature = 300
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
```

````{note}
Here we chose to create more refined mesh from the hydrogen transport model than was used in OpenMC. We could, however, use the same mesh from OpenMC by running:
```
my_model.mesh = F.Mesh(reader.dolfinx_mesh)
```
````

+++

Since we chose not to use the OpenMC mesh, we need to interpolate the results onto our new FESTIM mesh:

```{code-cell} ipython3
original_source_term = reader.create_dolfinx_function("tritium_production")
V = dolfinx.fem.functionspace(refined_mesh, ("DG", 0))
source_term = dolfinx.fem.Function(V)
F.helpers.nmm_interpolate(source_term, original_source_term)
```

Now we can add our *interpolated* tritium source into our problem:

```{code-cell} ipython3
tritium_source = F.ParticleSource(
    value=source_term, species=tritium, volume=volume
)

my_model.sources = [tritium_source]
```

Finally, let's solve and take a look at the results:

```{code-cell} ipython3
my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

from dolfinx import plot
import pyvista

c = tritium.post_processing_solution

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

As we'd expect, there is no diffusion on the outer boundaries of our cube (recall our insulated boundary conditions). Let's take a look inside the cube to how the tritium source diffuses inside the material:

```{code-cell} ipython3
:tags: [hide-input]

bounds = [0, dim, 0, dim, 0, dim]

clipped = u_grid.clip_box(bounds, invert=True)

pl = pv.Plotter()
pl.add_mesh(clipped, cmap="viridis", show_edges=False)
pl.view_xy()
pl.show()
```

We see a high concentration where the tritium is generated and it diffuses throughout the material. Success!
