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

# Concentration #

Boundary conditions (BCs) are essential to FESTIM simulations, as they describe the mathematical problem at the boundaries of the simulated domain. Read more about BCs _[here](https://festim.readthedocs.io/en/fenicsx/userguide/boundary_conditions.html)_.

This tutorial goes over how to define concentration boundary conditions for hydrogen transport simulations.

Objectives:
* Understand the mathematics behind a fixed concentration BC
* Define a fixed concentration BC
* Choose which solubility law (Sieverts' or Henry's)
* Solve a hydrogen transport problem with plasma implantation

+++

## Understanding mathematics behind a fixed concentration BC ##

A fixed concentration (Dirichlet) boundary condition prescribes the value of the mobile hydrogen isotope concentration at a boundary. This enforces the concentration to remain constant in time and space on the specified boundary, independent of the solution in the bulk.

This boundary condition is typically used to represent surfaces in equilibrium with an infinite reservoir, imposed implantation conditions, or experimentally controlled concentrations.

### Mathematical formulation

On a boundary $\Gamma_D$, the mobile concentration satisfies

$$
c(\mathbf{x}, t) = c_0 \quad \text{for } \mathbf{x} \in \Gamma_D,
$$

where $c_0$ is the prescribed concentration value.

In the weak formulation, this condition is enforced by directly constraining the degrees of freedom associated with the boundary.

+++

## Defining fixed concentration ##

Users can prescribe a fixed concentration on a boundary using `FixedConcentrationBC`, which can depend on temperature, time, and space.

Let us consider a 2D, steady-state, single-species example with the following *space-dependent* boundary conditions:

$$ \text{Top:} \quad c = 2x + y $$
$$ \text{Right:} \quad c = y^2 + 1 $$

```{code-cell} ipython3
import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

top_subdomain = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
right_subdomain = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1))
left_subdomain = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
bottom_subdomain = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[1], 0))

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(create_unit_square(MPI.COMM_WORLD, 10, 10))

H = F.Species("H")
my_model.species = [H]
mat = F.Material(D_0=1e-3, E_D=0)
my_model.material = mat
vol = F.VolumeSubdomain(id=5, material=mat)
my_model.subdomains = [top_subdomain, right_subdomain, left_subdomain, bottom_subdomain, vol]

my_model.temperature = 400
my_model.settings = F.Settings(atol=1e-8, rtol=1e-8, transient=False)
```

We can define functions for the top and right boundary conditions using `lambda` functions:

```{code-cell} ipython3
top_bc_function = lambda x: 2.0 * x[0] + x[1]
right_bc_function = lambda x: x[1] ** 2 + 1.0
```

<!-- ```{note}
    `x[0]`, `x[1]`, `x[2]` corresponse to *(x, y, z)* in cartesian space.
``` -->

+++

To include these boundary conditions to our problem, we use `FixedConcentrationBC`. We must also specify which subdomain (boundary) each BC is applied to, as well as the corresponding species:

```{code-cell} ipython3
top_bc = F.FixedConcentrationBC(
    subdomain=top_subdomain,
    value=top_bc_function,
    species=H
)

right_bc = F.FixedConcentrationBC(
    subdomain=right_subdomain,
    value=right_bc_function,
    species=H
)

# left_bc = F.FixedConcentrationBC(
#     subdomain=left_subdomain,
#     value=.0,
#     species=H  
# )

# bottom_bc = F.FixedConcentrationBC(
#     subdomain=bottom_subdomain,
#     value=0.0,
#     species=H  
# )   
```

Finally, we add our BCs to `my_model.boundary_conditions` using a list, and then run the model:

```{code-cell} ipython3
my_model.boundary_conditions = [top_bc, right_bc]
my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import pyvista
from dolfinx import plot

topology, cell_types, geometry = plot.vtk_mesh(H.post_processing_solution.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["c"] = H.post_processing_solution.x.array
grid.set_active_scalars("c")

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")

plotter = pyvista.Plotter()

plotter.add_mesh(grid)
plotter.view_xy()
contours = grid.contour(isosurfaces=5, scalars="c")
plotter.add_mesh(contours, color="r", line_width=0.1)
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("concentration.png")
```

```{note}

In this example, we did not define any boundary conditions for the left and bottom surfaces. If no BC is set on a boundary, it is assumed that the flux is null (symmetry boundary condition):

$$
\mathbf{J} \cdot \mathbf{n} = 0 \quad \text{on } \Gamma
$$

```

+++

### Time and temperature dependent boundary conditions ###

Users can also impose concentration BCs that are dependent on space, time and temperature, such as:

$$ c = 10 + x^2 + T(t+2) $$

To do so, we define a Python function:

```{code-cell} ipython3
my_custom_value = lambda x, t, T: 10 + x[0]**2 + T *(t + 2)

boundary = F.SurfaceSubdomain(id=1)
my_bc = F.FixedConcentrationBC(subdomain=boundary, value=my_custom_value, species=H)
```

## Choosing a solubility law ##

Users can define the surface concentration using either Sieverts’ law, $c = S(T)\sqrt P$, or Henry's law, $c=K_H P$, where $S(T)$ and $K_H$ denote the temperature-dependent Sieverts’ and Henry’s solubility coefficients, respectively, and $P$ is the partial pressure of the species on the surface. 

For Sieverts' law of solubility, we can use `festim.SievertsBC`:

```{code-cell} ipython3
from festim import SievertsBC, SurfaceSubdomain, Species

boundary = SurfaceSubdomain(id=1)
H = Species(name="Hydrogen")

custom_pressure_value = lambda t: 2 + t
my_bc = SievertsBC(subdomain=3, S_0=2, E_S=0.1, species=H, pressure=custom_pressure_value)
```

Similarly, for Henry's law of solubility, we can use `festim.HenrysBC`:

```{code-cell} ipython3
from festim import HenrysBC

pressure_value = lambda t: 5 * t
my_bc = HenrysBC(subdomain=3, H_0=1.5, E_H=0.2, species=H, pressure=pressure_value)
```

## Plasma implantation approximation ##

+++

We can also approximate plasma implantation using FESTIM's `ParticleSource` class, which is helpful in modeling thermo-desorption spectra (TDS) experiments or including the effect of plasma exposure on hydrogen transport. Learn more about how FESTIM approximates plasma implantation _[here](https://festim.readthedocs.io/en/fenicsx/theory.html)_.

Consider the following 1D plasma implantation problem, where we represent the plasma as a hydrogen source $S_{ext}$:

$$ S_{ext} = \varphi \cdot f(x) $$

$$\varphi = 1\cdot 10^{13} \quad \mathrm{m}^{-2}\mathrm{s}^{-1}$$

where  $\varphi$ is the implantation flux and $f(x)$ is a Gaussian spatial distribution (distribution mean value of 0.5 $\text{m}$ and width of 1 $\text{m}$).

First, we setup a 1D mesh ranging from $ [0,1] $ and assign the subdomains and material:

```{code-cell} ipython3
import festim as F
import ufl
import numpy as np

my_model = F.HydrogenTransportProblem()
vertices = np.linspace(0,1,2000)
my_model.mesh = F.Mesh1D(vertices)

mat = F.Material(D_0=0.1, E_D=0.01)

volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
right_boundary = F.SurfaceSubdomain1D(id=2, x=1)

my_model.subdomains = [
    volume_subdomain,
    left_boundary,
    right_boundary,
]
```

Now, we define our `incident_flux` and `gaussian_distribution` function. We can use `ParticleSource` to represent the source term, and then add it to our model:

```{code-cell} ipython3
incident_flux = 1e13 

def gaussian_distribution(x, center, width):
    return (
        1
        / (width * (2 * ufl.pi) ** 0.5)
        * ufl.exp(-0.5 * ((x[0] - center) / width) ** 2)
    )
H = F.Species("H")
my_model.species = [H]

source_term = F.ParticleSource(
    value=lambda x: incident_flux * gaussian_distribution(x, .5, 1),
    volume=volume_subdomain,
    species=H,
)

my_model.sources = [source_term]
```

Finally, we assign boundary conditions (zero concentration at $x=0$ and $x=1$) and solve our problem:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H),
    F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
]

my_model.temperature = 400
my_model.settings = F.Settings(atol=1e10, rtol=1e-10, transient=False)

profile_export = F.Profile1DExport(field=H,subdomain=volume_subdomain)
my_model.exports = [profile_export]

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

x = my_model.exports[0].x
c = my_model.exports[0].data[0][0]

plt.plot(x, c)
plt.show()
```
