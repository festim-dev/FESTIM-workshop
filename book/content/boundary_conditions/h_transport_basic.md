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

# Hydrogen transport: basic

This section discusses how to implement basic boundary conditions (BCs) for hydrogen transport problems. Boundary conditions are essential to FESTIM simulations, as they describe the mathematical problem at the boundaries of the simulated domain. Read more about BCs _[here](https://festim.readthedocs.io/en/fenicsx/userguide/boundary_conditions.html)_.

Objectives:
* Learn mathematical formulation for concentration and flux boundary conditions
* Implement fixed concentration boundary conditions
* Implement particle flux boundary conditions

+++

## Understanding math behind concentration and flux boundary conditions

### Fixed concentration

A fixed concentration (Dirichlet) boundary condition prescribes the value of the mobile hydrogen isotope concentration at a boundary. This enforces the concentration to remain constant in time and space on the specified boundary, independent of the solution in the bulk.

This boundary condition is typically used to represent surfaces in equilibrium with an infinite reservoir, imposed implantation conditions, or experimentally controlled concentrations.

#### Mathematical formulation

On a boundary $\Gamma_D$, the mobile concentration satisfies

$$
c(\mathbf{x}, t) = c_0 \quad \text{for } \mathbf{x} \in \Gamma_D,
$$

where $c_0$ is the prescribed concentration value.

In the weak formulation, this condition is enforced by directly constraining the degrees of freedom associated with the boundary.

### Particle flux boundary conditions

A particle flux (Neumann) boundary condition prescribes the normal flux of mobile hydrogen isotopes across a boundary. Unlike fixed concentration conditions, flux boundary conditions do not directly constrain the concentration value at the surface. Instead, they control the rate at which particles enter or leave the domain.

Flux boundary conditions are commonly used to represent implantation from a plasma, outgassing to vacuum, permeation through a surface, or symmetry boundaries where no net transport occurs.

#### Mathematical formulation

The particle flux $\mathbf{J}$ is typically given by Fick’s law,

$$
\mathbf{J} = -D \nabla c,
$$

where $D$ is the diffusion coefficient and $c$ is the mobile concentration.

On a boundary $\Gamma_N$, a prescribed normal flux is imposed as

$$
\mathbf{J} \cdot \mathbf{n} = g(\mathbf{x}, t) \quad \text{for } \mathbf{x} \in \Gamma_N,
$$

where:
- $\mathbf{n}$ is the outward unit normal vector,
- $g(\mathbf{x}, t)$ is the imposed particle flux (positive for outward flux, negative for inward flux).

A special case is the **zero-flux (symmetry) boundary condition**, given by

$$
\mathbf{J} \cdot \mathbf{n} = 0 \quad \text{on } \Gamma,
$$

which implies no net particle transport across the boundary.

#### Weak formulation

In the weak form, flux boundary conditions appear naturally as surface integrals after integration by parts. For a test function $v$, the boundary contribution is

$$
\int_{\Gamma_N} g(\mathbf{x}, t)\, v \, \mathrm{d}\Gamma.
$$

Because of this, flux boundary conditions are often referred to as *natural boundary conditions* and do not require explicit modification of the solution space, unlike Dirichlet conditions.

+++

## Imposing fixed concentration boundary conditions

Users can prescribe a fixed concentration on a boundary using `FixedConcentrationBC`, which can depend on temperature, time, and space.

Let us consider a 2D, steady-state, single-species example with the following *space-dependent* boundary conditions:

$$ \text{Top:} \quad c = 2x + y $$
$$ \text{Right:} \quad c = y^2 + 1 $$

```{code-cell} ipython3
:tags: [hide-input]

import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(create_unit_square(MPI.COMM_WORLD, 10, 10))
H = F.Species("H")
my_model.species = [H]
mat = F.Material(D_0=1e-3, E_D=0)
my_model.material = mat
my_model.temperature = 400
my_model.settings = F.Settings(atol=1e-8, rtol=1e-8, transient=False)
```

We can define the top and right boundary conditions using `lambda` functions:

```{code-cell} ipython3
top_bc_function = lambda x: 2.0 * x[0] + x[1]
right_bc_function = lambda x: x[1] ** 2 + 1.0
```

```{note}
`x[0]`, `x[1]`, `x[2]` corresponse to *(x, y, z)* in cartesian space. Additionally, boundary conditions can also be given as Python functions (see *{ref}`label-time-temp-concentration`* to learn more.)
```

+++

Now, let's create the boundaries to assign the BCs:

```{code-cell} ipython3
top_subdomain = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
right_subdomain = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1))
left_subdomain = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))
bottom_subdomain = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[1], 0))
vol = F.VolumeSubdomain(id=5, material=mat)

my_model.subdomains = [top_subdomain, right_subdomain, left_subdomain, bottom_subdomain, vol]
```

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
```

We can add these to our problem, we pass them into `boundary_conditions` as a list and then run the simulation:

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

pyvista.set_jupyter_backend("html")

plotter = pyvista.Plotter()

plotter.add_mesh(grid)
plotter.view_xy()
contours = grid.contour(isosurfaces=5, scalars="c")
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

(label-time-temp-concentration)=
### Adding time and temperature dependent boundary conditions ###

+++

Users can also impose concentration BCs that are dependent on space $x$, time $t$, and temperature $T$, such as:

$$ c = 10 + x^2 + T(t+2) $$

To do so, we define a Python function:

```{code-cell} ipython3
def my_custom_value(x, t, T):
    return 10 + x[0]**2 + T *(t + 2)

boundary = F.SurfaceSubdomain(id=1)
my_bc = F.FixedConcentrationBC(subdomain=boundary, value=my_custom_value, species=H)
```

```{tip}
This custom function is equivalent to:

`
my_custom_value = lambda x, t, T: 10 + x[0]**2 + T *(t + 2)
`
```

### Multi-species fixed concentration BC ##

Users may also want to add BCs for multiple species in their model. 

For example, if we wanted to impose separate fixed concentrations for deuterium and hydrogen on a boundary, we need to define each species and then one BC for each:

```{code-cell} ipython3
import festim as F

H = F.Species("H")
D = F.Species("D")
boundary = F.SurfaceSubdomain(id=1)

top_H = F.FixedConcentrationBC(
    subdomain=boundary,
    value=5.0,
    species=H
)

top_D = F.FixedConcentrationBC(
    subdomain=boundary,
    value=10.0,
    species=D
)       
```

## Imposing particle flux boundary conditions

+++

Users can impose the particle flux at boundaries using `ParticleFluxBC` class. At minimum, this BC requires a boundary, value, and species to be added:

```{code-cell} ipython3
import festim as F

boundary = F.SurfaceSubdomain(id=1)
H = F.Species(name="H")

my_flux_bc = F.ParticleFluxBC(subdomain=boundary, value=2, species=H)
```

To use this BC to your problem, simply add it to `boundary_conditions` attribute of your problem:

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
my_model.boundary_conditions = [my_flux_bc]
```

### Species-dependent fluxes

Similar to the concentration, the flux can be dependent on space, time and temperature. But for particle fluxes, the values can also be dependent on a species’ concentration. 

For example, let's define a hydrogen flux `J` that depends on the hydrogen concentration `c` and time `t`:

$$ J(c,t) = 10t^2 + 2c $$

```{code-cell} ipython3
import festim as F

my_model = F.HydrogenTransportProblem()
boundary = F.SurfaceSubdomain(id=1)
H = F.Species(name="H")

J = lambda t, c: 10*t**2 + 2*c
```

To add this BC, we need to create a dictionary that maps the concentration variable `c` in our custom function to our species `H`:

```{code-cell} ipython3
mapping_dict = {"c": H}
```

Now, we add our `ParticleFluxBC`, also utilizing the `species_dependent_value` argument:

```{code-cell} ipython3
my_flux_bc = F.ParticleFluxBC(
    subdomain=boundary,
    value=J,
    species=H,
    species_dependent_value=mapping_dict,
)

my_model.boundary_conditions = [my_flux_bc]
```

### Multi-species flux boundary conditions

Users may also need to impose a flux boundary condition which depends on the concentrations of multiple species. 

Consider the following 1D example with three species, $\mathrm{A}$, $\mathrm{B}$, and $\mathrm{C}$. The boundary conditions include recombination fluxes $\phi_{AB}$ and $\phi_{ABC}$ that depend on the concentrations $\mathrm{c}$ (on the right) and fixed concentrations (on the left):

$$ \text{Right (recombination flux):} \quad \phi_{AB} = -c_A c_B $$

$$ \text{Right (recombination flux):} \quad \phi_{ABC} = -10 c_A c_B c_C$$

$$ \text{Left (fixed concentration):} \quad c_A = c_B = c_C = 1 $$


First, let us define each species using `Species`:

```{code-cell} ipython3
import festim as F

my_model = F.HydrogenTransportProblem()

A = F.Species(name="A")
B = F.Species(name="B")
C = F.Species(name="C")

my_model.species = [A, B, C]
```

Now, we create the dictionary to be passed into `species_dependent_value`; this maps each argument in the custom flux function to the corresponding defined species:

```{code-cell} ipython3
species_dependent_value = {"c_A": A, "c_B": B, "c_C": C}
```

We also define our custom functions for the fluxes:

```{code-cell} ipython3
recombination_flux_AB = lambda c_A, c_B, c_C: -c_A*c_B
recombination_flux_ABC = lambda c_A, c_B, c_C: -10*c_A*c_B*c_C
```

Now, we create our 1D mesh and corresponding boundaries:

```{code-cell} ipython3
import numpy as np

my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

D = 1e-2
mat = F.Material(
    D_0={A: 8*D, B: 7*D, C: D},
    E_D={A: 0.01, B: 0.01, C: 0.01},
)

bulk = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
left = F.SurfaceSubdomain1D(id=1, x=0)
right = F.SurfaceSubdomain1D(id=2, x=1)
my_model.subdomains = [bulk, left, right]
```

```{note}
The diffusivity pre-factor `D_0` and activation energy `E_D` must be defined for each species in `Material`. Learn more about defining multi-species material properties [here](../material/material_advanced.md). Here we choose arbitrarily different diffusivity coefficients to see transport between species. 
```

+++

Let's assign boundary conditions (recombination on the right, fixed concentration on the left). The boundary condition `ParticleFluxBC` must be added for each species:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.ParticleFluxBC(
        subdomain=right,
        value=recombination_flux_AB,
        species=A,
        species_dependent_value=species_dependent_value,
    ),
    F.ParticleFluxBC(
        subdomain=right,
        value=recombination_flux_AB,
        species=B,
        species_dependent_value=species_dependent_value,
    ),
    F.ParticleFluxBC(
        subdomain=right,
        value=recombination_flux_ABC,
        species=A,
        species_dependent_value=species_dependent_value,
    ),
    F.ParticleFluxBC(
        subdomain=right,
        value=recombination_flux_ABC,
        species=B,
        species_dependent_value=species_dependent_value,
    ),
    F.ParticleFluxBC(
        subdomain=right,
        value=recombination_flux_ABC,
        species=C,
        species_dependent_value=species_dependent_value,
    ),
    F.FixedConcentrationBC(subdomain=left,value=1,species=A),
    F.FixedConcentrationBC(subdomain=left,value=1,species=B),
    F.FixedConcentrationBC(subdomain=left,value=1,species=C),     
]
```

We can export the flux for each species on the right using `SurfaceFlux` (see [post-processing derived values](../post_process/derived.md) to learn more about exporting fluxes):

```{code-cell} ipython3
right_flux_A = F.SurfaceFlux(field=A,surface=right)
right_flux_B = F.SurfaceFlux(field=B,surface=right)
right_flux_C = F.SurfaceFlux(field=C,surface=right)

my_model.exports = [
    right_flux_A,
    right_flux_B,
    right_flux_C,
]
```

Finally, let's solve and plot the profile for each species:

```{code-cell} ipython3
:tags: [hide-input]

my_model.temperature = 300
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()

import matplotlib.pyplot as plt

def plot_profile(species, **kwargs):
    c = species.post_processing_solution.x.array[:]
    x = species.post_processing_solution.function_space.mesh.geometry.x[:,0]
    return plt.plot(x, c, **kwargs)

for species in my_model.species:
    plot_profile(species, label=species.name)

plt.xlabel('Position')
plt.ylabel('Concentration')
plt.legend()
plt.show()
```

We see the higher recombination flux for $\mathrm{ABC}$ decreases the concentration of $\mathrm{C}$ throughout the material.
