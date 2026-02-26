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

# Exporting fields #

FESTIM has convenience classes that allow users to create XDMF and VTX exports, which can then be viewed in Paraview.

Objectives:
* Writing VTX files for species and temperature fields
* Additional options for VTX exports
* Writing XDMF files for field exports

+++

## Writing VTX files for species and temperature fields ##

Users can export concentration fields to VTX using `VTXSpeciesExport`, and then view their exports in ParaView. This example will discuss how to define the export, and what result should you expect to see.

Let us setup a 2D, transient problem with the following boundary conditions:

$$ \text{Right surface:} \quad c_H = 0 $$
$$ \text{Left surface:} \quad c_H = 1 $$

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import festim as F
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(mesh)
mat = F.Material(D_0=1e-2, E_D=0)

right_surface = F.SurfaceSubdomain(id=1, locator = lambda x: np.isclose(x[0], 1.0))
left_surface = F.SurfaceSubdomain(id=2, locator = lambda x: np.isclose(x[0], 0.0))
vol = F.VolumeSubdomain(id=1, material=mat)

H = F.Species("H")

my_model.subdomains = [right_surface, left_surface, vol]
my_model.species = [H]
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=right_surface, value=0, species=H),
    F.FixedConcentrationBC(subdomain=left_surface, value=1, species=H)
]
my_model.temperature = lambda x: 400 + 10*x[0] + 2*x[1]
my_model.settings = F.Settings(atol=1e-10,rtol=1e-10,stepsize=1, final_time=10)
```

We can export the concentration field by defining a `VTXSpeciesExport` object for our model. The required arguments are a filename path (which must end in `.bp`) and the field (species) you want to export:

```{code-cell} ipython3
H_export = F.VTXSpeciesExport(filename="H_concentration.bp", field=H)
```

```{note}
If we had several `Species`, we would create several `VTXSpeciesExport` objects, one per species.
```

If needed, we can also export the temperature field this way using `VTXTemperatureExport`

```{code-cell} ipython3
temp_export = F.VTXTemperatureExport(filename="temperature.bp")
```

Then, we just pass these exports to `my_model.exports` as a list:

```{code-cell} ipython3
my_model.exports = [H_export, temp_export]

my_model.initialise()
my_model.run()
```

We should expect to see two new folders called `H_concentration.bp` and `temperature.bp`. To view the results, we can use ParaView (see the [ParaView section](paraview.md) to learn more).

+++

## Exporting results for a discontinuous problem ##
If running a multi-material discontinuous simulation, it is necessary to also specify the subdomain during the export process.

Consider the same [multi-material problem](../material/material_basics.md) in the Materials chapter, where we use `HydrogenTransportProblemDiscontinuous` to solve a multi-material problem:

```{code-cell} ipython3
:tags: [hide-input]

import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
festim_mesh = F.Mesh(fenics_mesh)

my_model = F.HydrogenTransportProblemDiscontinuous()

material_top = F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0)
material_bottom = F.Material(D_0=2, E_D=0, K_S_0=3, E_K_S=0)

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
H.subdomains = [top_volume, bottom_volume]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
```

We can specify separate export objects for the top and bottom domains using the `subdomain` argument, and should expect to see two new folders created named `top.bp` and `bottom.bp`:

```{code-cell} ipython3
top_export = F.VTXSpeciesExport(filename="top.bp", field=H, subdomain=top_volume)
bottom_export = F.VTXSpeciesExport(filename="bottom.bp", field=H, subdomain=bottom_volume)
my_model.exports = [
    top_export,
    bottom_export,
]
my_model.initialise()
my_model.run()
```

Finally, these `.bp` exports can be viewed in ParaView:

+++

```{image} multi_material_paraview.png
:alt: multi
:class: bg-primary mb-1
:align: center
```

```{note}
For multi-material discontinuous problems, each .bp file only shows its corresponding subdomain.
```

+++

## Exporting fields at specific timesteps ##
Users can also specify which timesteps they'd like to export using the `times` argument (which must be a list):

```{code-cell} ipython3
export = F.VTXSpeciesExport(filename="H_concentration.bp", field=H, times=[0, 5, 10])
```

If no `times` argument is given, the export stores results for all timesteps by default.

+++

(checkpointing)=
## Checkpointing ##

It may be helpful to store results from one simulation for later use in another (perhaps as an initial condition, see [](ic-checkpoint)). FESTIM includes this capability by incorporationg `adios4dolfinx` functionality, which stores mesh information and solutions into a `checkpoint.bp` file. Learn more about [checkpointing in DOLFINx here](https://jsdokken.com/adios4dolfinx/README.html).

To store the species field as a checkpoint file, simply set the `checkpoint` argument to `True`:

```{code-cell} ipython3
export = F.VTXSpeciesExport(filename="H_concentration.bp", field=H, checkpoint=True)
```

```{Note}
Checkpointed files cannot be viewed in ParaView.
```

+++

## Writing XDMF files for field exports ##

```{warning}
Exporting to VTX is preferable over XDMF, as XDMF functionality will soon be deprecated. Additionally, you cannot view transient results using XDMF.
```

Users can export functions to XDMF files using the `XDMFExport` class, which requires a `filename` and `field`:

```{code-cell} ipython3
import festim as F

H = F.Species("H")
export = F.XDMFExport(filename="my_export.xdmf", field=H)
```

To export this in a FESTIM simulation, add the export to your problem's `export` attribute:

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
my_model.exports = [export]
```

This will produce the corresponding export files (`my_export.xdmf` and `my_export.h5`).

+++

