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

# Particle flux #

This tutorial goes over how to define particle flux boundary conditions in FESTIM.

Objectives:
* Learn how to define fluxes at boundaries
* Impose boundary fluxes as a function of the concentration
* Solve a problem with multi-species flux boundary conditions

+++

## Defining flux at boundaries ##

Users can impose the particle flux (Neumann BC) at boundaries using `ParticleFluxBC` class:

```{code-cell} ipython3
from festim import ParticleFluxBC, Species, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
H = Species(name="Hydrogen")

my_flux_bc = ParticleFluxBC(subdomain=boundary, value=2, species=H)
```

## Defining concentration-dependant fluxes ##

Similar to the concentration, the flux can be dependent on space, time and temperature. But for particle fluxes, the values can also be dependent on a species’ concentration. 

For example, let's define a hydrogen flux `J` that depends on the hydrogen concentration `c` and time `t`:

$$ J(c,t) = 10t^2 + 2c $$

```{code-cell} ipython3
from festim import ParticleFluxBC, Species, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
H = Species(name="Hydrogen")

J = lambda t, c: 10*t**2 + 2*c

my_flux_bc = ParticleFluxBC(
    subdomain=boundary,
    value=J,
    species=H,
    species_dependent_value={"c": H},
)
```

## Multi-species flux boundary conditions ##

+++

Users may also need to impose a flux boundary condition in multi-species problems where the flux depends on the concentrations of multiple species. 

Consider the following example with three species, A, B, and C, where the particle flux boundary condition depends on each species' concentration:

$$ J(c_A, c_B, c_C) = 2c_A + 3c_b + 4c_C $$

We must first define each species using `Species` and then create the dictionary to be passed into `species_dependent_value`. The dictionary maps each argument in the custom flux function to the corresponding defined species:

```{code-cell} ipython3
import festim as F

my_model = F.HydrogenTransportProblem()

A = F.Species(name="A")
B = F.Species(name="B")
C = F.Species(name="C")
my_model.species = [A, B, C]

my_custom_value = lambda c_A, c_B, c_C: 2*c_A + 3*c_B + 4*c_C
species_dependent_value = {"c_A": A, "c_B": B, "c_C": C}
```

Now, we create our 1D mesh and assign boundary conditions (flux BC on the left). The boundary condition `ParticleFluxBC` must be added for each species:

```{code-cell} ipython3
import numpy as np

my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

D = 1
mat = F.Material(
    D_0={A: D, B: D, C: D},
    E_D={A: 0.01, B: 0.01, C: 0.01},
)

bulk = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
left = F.SurfaceSubdomain1D(id=1, x=0)
right = F.SurfaceSubdomain1D(id=2, x=1)

my_model.subdomains = [bulk, left, right]
my_model.boundary_conditions = [
    F.ParticleFluxBC(
        subdomain=left,
        value=my_custom_value,
        species=A,
        species_dependent_value=species_dependent_value,
    ),
    F.ParticleFluxBC(
        subdomain=left,
        value=my_custom_value,
        species=B,
        species_dependent_value=species_dependent_value,
    ),
    F.ParticleFluxBC(
        subdomain=left,
        value=my_custom_value,
        species=C,
        species_dependent_value=species_dependent_value,
    ),
]
```

```{note}
The diffusivity pre-factor `D_0` and activation energy `E_d` must be defined for each species in `Material`. Learn more about defining multi-species material properties __[here](https://festim-workshop.readthedocs.io/en/festim2/content/material/material_advanced.html#defining-species-dependent-material-properties)__. 
```

+++

Finally, let's solve and plot the solution for each species:

```{code-cell} ipython3
my_model.temperature = 300
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

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
