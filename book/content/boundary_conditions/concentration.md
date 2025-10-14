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
* Define a fixed concentration BC
* Choose which solubility law (Sieverts' or Henry's)
* Solve a hydrogen transport problem with plasma implantation

+++

## Defining fixed concentration ##

BCs need to be assigned to surfaces using FESTIM's `SurfaceSubdomain` class. To define the concentration of a specific species, we can use `FixedConcentrationBC`:

```{code-cell} ipython3
from festim import FixedConcentrationBC, Species, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
H = Species(name="Hydrogen")

my_bc = FixedConcentrationBC(subdomain=boundary, value=10, species=H)
```

The imposed concentration can be dependent on space, time and temperature, such as:

$$ 

BC = 10 + x^2 + T(t+2)

$$

```{code-cell} ipython3
my_custom_value = lambda x, t, T: 10 + x[0]**2 + T *(t + 2)

my_bc = FixedConcentrationBC(subdomain=boundary, value=my_custom_value, species=H)
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
