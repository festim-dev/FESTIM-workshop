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

# Field exports #

FESTIM has classes to creae XDMF and VTX exports, which can then be viewed in Paraview (see our intro to Paraview tutorial here)!

Objectives:
* Writing XDMF files
* Writing VTX files
* Exporting fields in a discontinuous problem

+++

## Writing XDMF files ##

Users can export functions to XDMF files using the `XDMFExport` class, which requires a `filename` and `field`:

```{code-cell} ipython3
import festim as F

H = F.Species("H")
my_export = F.XDMFExport(filename="my_export.xdmf", field=H)
```

## Writing VTK files ##

Users can also export temperature and concentration fields to VTX files using `VTXTemperatureExport` and `VTKSpeciesExport`, respectively. For both classes, we need to provide the `filename` for the output and an optional list of `times` (exports all times otherwise, defaults to `None`).

To export a temperature field using `VTXTemperatureExport`:

```{code-cell} ipython3
import festim as F

my_model = F.HydrogenTransportProblem()
my_model.exports = [
    F.VTXTemperatureExport(filename="out.bp",times=[0, 5, 10])
]
```

Exporting the concentration also requires us to define the `field` to export, which `subdomain` to export on (defaults to all if none is provided), and the option to turn on `checkpoints` (exports to a checkpoint file using __[adios4dolfinx](https://github.com/jorgensd/adios4dolfinx)__) (defaults to `False`):

```{code-cell} ipython3
H = F.Species("H")
subdomain = F.SurfaceSubdomain(id=1)
my_model.species = [H]
my_model.exports = [
    F.VTXSpeciesExport(filename="out.bp",field=H,subdomain=subdomain,checkpoint=False)
]
```

## Exporting fields in a discontinuous problem ##

Consider the same __[multi-material problem](https://festim-workshop.readthedocs.io/en/festim2/content/material/material_basics.html#multi-material-example)__ in the Materials chapter, where we use `HydrogenTransportProblemDiscontinuous` to solve a multi-material problem:

```{code-cell} ipython3
import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
festim_mesh = F.Mesh(fenics_mesh)

my_model = F.HydrogenTransportProblemDiscontinuous()

material_top = F.Material(
    D_0=1,    
    E_D=0,     
    K_S_0=2,  
    E_K_S=0,     
)

material_bottom = F.Material(
    D_0=2,    
    E_D=0,     
    K_S_0=3,    
    E_K_S=0,     
)

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

Instead of viewing the results using PyVista, we can export the fields for each subdomain in Paraview using `VTXSpeciesExport`:

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

```{image} multi_material_paraview.png
:alt: multi
:class: bg-primary mb-1
:align: center
```

```{note}
For multi-material discontinuous problems, exports need to be written to VTX files, not XDMF.
```
