---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

# Derived quantities #

This tutorial goes over FESTIM's built-in functions to help users export derived results.

Objectives:
* Exporting surface quantities
* Exporting volume quantities
* Exporting multiple derived quantities

+++

## Exporting surface quantities ##

Users can export surface values in FESTIM by passing in the desired `field`, `SurfaceSubdomain`, and an optional `filename` (ending in `.txt` or `.csv`).

To get total, average, minimum, and maximum surface values, we can use the `TotalSurface`, `AverageSurface`, `MinimumSurface`, and `MaximumSurface` classes:

```{code-cell} ipython3
import festim as F

left = F.SurfaceSubdomain(1)
H = F.Species("H")

my_total = F.TotalSurface(field=H, surface=left,filename="total.csv")
my_average = F.AverageSurface(field=H, surface=left,filename="avg.csv")
my_minimum = F.MinimumSurface(field=H, surface=left,filename="min.csv")
my_maximum = F.MaximumSurface(field=H, surface=left,filename="max.csv")
```

These exports will result in a text file with a list of data points and corresponding time steps. General surface quantities can be exported using `SurfaceQuantity`.

We can also calculate the surface flux using the `SurfaceFlux` class:

```{code-cell} ipython3
my_flux = F.SurfaceFlux(field=H, surface=left,filename="flux.csv")
```

## Exporting volume quantities ##

Volume quanities can similary be exported in FESTIM, except now we must pass in a `VolumeSubdomain`.

To get total, average, minimum, and maximum volume values, we can use the `TotalVolume`, `AverageVolume`, `MinimumVolume`, and `MaximumVolume` classes:

```{code-cell} ipython3
import festim as F

mat = F.Material(D_0=1, E_D=0)
vol = F.VolumeSubdomain(1,material=mat)
H = F.Species("H")

my_total = F.TotalVolume(field=H, volume=vol,filename="total.csv")
my_average = F.AverageVolume(field=H, volume=vol,filename="avg.csv")
my_minimum = F.MinimumVolume(field=H, volume=vol,filename="min.csv")
my_maximum = F.MaximumVolume(field=H, volume=vol,filename="max.csv")
```

General volume quantities can be exported using `VolumeQuantity`.

+++

## Exporting multiple derived quanitites ##

To export your results from a FESTIM simulation, you can pass a list of derived quantity objects into the `export` attribute for your problem:

```{code-cell} ipython3
import numpy as np
import festim as F
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(mesh)
mat = F.Material(D_0=1, E_D=0)
right_surface = F.SurfaceSubdomain(id=1, locator = lambda x: np.isclose(x[0], 1.0))
left_surface = F.SurfaceSubdomain(id=2, locator = lambda x: np.isclose(x[0], 0.0))
vol = F.VolumeSubdomain(id=1, material=mat)
H = F.Species("H")

max_surface = F.MaximumSurface(field=H, surface=left_surface)
avg_vol = F.AverageVolume(field=H, volume=vol)

my_model.subdomains = [right_surface, left_surface, vol]
my_model.species = [H]
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_surface, value=1, species=H)
]
my_model.temperature = 400
my_model.settings = F.Settings(atol=1e10,rtol=1e-10,transient=False)
my_model.exports = [
    max_surface,
    avg_vol
]

my_model.initialise()
my_model.run()
```
