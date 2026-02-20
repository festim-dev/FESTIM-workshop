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

Objectives:
* Understanding the importance of fluid flow in hydrogen diffusion
* Utilizing OpenFOAM to generate velocity fields
* Reading fields into FESTIM for advective-assisted diffusion

+++

## Importance of fluid flow in hydrogen diffusion

Let's review the governing equation for FESTIM, neglecting reactions and source terms:

$$
\frac{\partial c_m}{\partial t} = \nabla \cdot (D \nabla c_m) + \mathbf{u} \cdot \nabla c_m ,
$$

where $\mathbf{u}$ is the velocity field in units of $\mathrm{ms^{-1}}$. This velocity term, which represents the species' advection, requires solving for the $\textbf{u}$ field. This can be done using custom 

Additionally, turbulence assists diffusion:

+++

### When do I need advection in my diffusion model? 

Discuss Peclet number

+++

## Solving a lid-driven cavity problem

This section discusses how to setup and solve a coupled problem between OpenFOAM and FESTIM, as outlined in the FESTIM 2 review paper (see [here](https://arxiv.org/abs/2509.24760) to learn more). Specifically, we'll solve a lid-driven cavity problem by calculating the velocity field in OpenFOAM and exporting it to FESTIM to solve our diffusion model.


We'l; 

$$
\mathbf{u}(x=0, y) = (0, 0), \quad
\mathbf{u}(x=L, y) = (0, 0), \\
\mathbf{u}(x, y=0) = (0, 0), \quad
\mathbf{u}(x, y=L) = (U, 0)
$$

+++

First, let's initiate our problem and create our rectanglar mesh:

```{code-cell} ipython3
import festim as F
from mpi4py import MPI
from dolfinx.mesh import create_rectangle
import numpy as np
from pathlib import Path
from dolfinx import fem
import zipfile
from dolfinx.io import VTXWriter

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

# define sources
my_model.sources = [
    F.ParticleSource(volume=fluid, species=H, value=10),
]

```
