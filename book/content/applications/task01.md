---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

(simple-simulation)=
# Simple simulation

In this task, we'll go through the basics of FESTIM and run a simple simulation on a 1D domain.

The very first step is to import the {py:mod}`festim<festim>` package:

```{code-cell} ipython3
import festim as F

print(F.__version__)
```

We then create a {py:class}`HydrogenTransportProblem<festim.hydrogen_transport_problem.HydrogenTransportProblem>` object.

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
```

## Mesh

FESTIM simulations need a mesh, here we use {py:class}`Mesh1D<festim.mesh.Mesh1D>`.

```{code-cell} ipython3
import numpy as np

my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, num=1001))
```

```{seealso}
For more information on meshes in FESTIM, see [](meshes).
```

+++

## Materials

{py:class}`Material<festim.material.Material>` objects hold the materials properties like diffusivity.

```{code-cell} ipython3
mat = F.Material(D_0=1, E_D=0.0)

volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
boundary_left = F.SurfaceSubdomain1D(id=1, x=0)
boundary_right = F.SurfaceSubdomain1D(id=2, x=1)
my_model.subdomains = [volume_subdomain, boundary_left, boundary_right]
```

```{code-cell} ipython3
H = F.Species("H")
my_model.species = [H]
```

## Temperature

```{code-cell} ipython3
my_model.temperature = 300
```

## Boundary conditions

Our hydrogen transport problem now needs boundary conditions.

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=boundary_left, value=1, species=H),
    F.FixedConcentrationBC(subdomain=boundary_right, value=0, species=H),
]
```

## Settings

With {py:class}`Settings<festim.settings.Settings>` we set the main solver parameters.

```{code-cell} ipython3
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=2)
```

## Stepsize

Since we are solving a transient problem, we need to set a {py:class}``Stepsize<festim.stepsize.Stepsize>``.
Here, the value of the stepsize is fixed at 0.05.

```{code-cell} ipython3
my_model.settings.stepsize = F.Stepsize(0.05)  # s
```

## Exports

Finally, we want to be able to visualise the concentration field.

```{code-cell} ipython3
profile = F.Profile1DExport(field=H, subdomain=volume_subdomain, times=[0.05, 0.1, 0.2, 0.5, 1])

my_model.exports = [
    F.VTXSpeciesExport(
        field=H,
        filename="hydrogen_concentration.bp",
    ),
    profile,
]
```

## Run

Finally, we initialise the model and run it!

```{code-cell} ipython3
my_model.initialise()

my_model.run()
```

You should now see the file `hydrogen_concentration.bp`.

```{seealso}
Check out the [Paraview](paraview) section for visualisation!
```

+++

## Visualise

```{code-cell} ipython3
import matplotlib.pyplot as plt

x = my_model.mesh.mesh.geometry.x[:, 0]
for time, data in zip(profile.t, profile.data):
    plt.plot(x, data, label=f"{time:.2f} s")

plt.xlabel("x (m)")
plt.ylabel("Mobile concentration (H/m3)")
plt.legend()
plt.show() 
```
