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

# Viewing exports in Paraview #

Paraview is a strong visualization tool that users can use to view their FESTIM exports. This tutorial goes over a simple introduction to viewing results in Paraview. Look at __[Paraview's download page](https://www.paraview.org/download/)__ for more information on installing Paraview.

Objectives
* Creating a VTX species export
* Opening file in Paraview

+++

## Creating a VTX species export  ##

First, let us run a simple 2D diffusion problem and export a VTK file:

```{code-cell} ipython3
import festim as F
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
festim_mesh = F.Mesh(fenics_mesh)

my_model = F.HydrogenTransportProblem()

mat = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain(id=1, material=mat)
top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))

my_model.mesh = festim_mesh
my_model.subdomains = [top_surface, bottom_surface, vol]

H = F.Species("H")
my_model.species = [H]
my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.exports = [
    F.VTXSpeciesExport(filename="paraview/out.bp",field=H,subdomain=vol,checkpoint=False)
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

## Opening file in Paraview ##

Now, we can open Paraview and open our exported file.

First, we need to select the correct file by navigating to the file browser located in the top left:

```{image} paraview/paraview_opening.png
:class: bg-primary mb-1
:align: center
```

Then, we select our `out.bp` file and press OK on the bottom:

```{image} paraview/paraview_selecting_file.png
:class: bg-primary mb-1
:align: center
```

Now we must press apply to visualize our domain:

```{image} paraview/paraview_select_apply.png
:class: bg-primary mb-1
:align: center
```

To view our concentration, we need to select the dropdown box that says "Solid Color" and change it to our field variable (named "f" in our example):

```{image} paraview/paraview_changing_variables.png
:class: bg-primary mb-1
:align: center
```

```{note}
Users can also choose to view other exported fields (such as temperature) using this dropdown section.
```

Finally, we see our diffusion field!
```{image} paraview/paraview_result.png
:class: bg-primary mb-1
:align: center
```
