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

(meshes)=
# Meshes

Every FESTIM simulation is solved on a mesh: the discretisation of your domain into cells. The mesh decides what geometry you can represent, and how finely you resolve steep gradients like the concentration profile just under an implanted surface.

FESTIM can build simple 1D meshes itself. For anything else you build the mesh elsewhere and hand it to FESTIM, so most of this section is about the "elsewhere".

```{admonition} Objectives
:class: objectives

* Choose the right meshing tool for your geometry
* Build 1D meshes directly in FESTIM, with refinement where you need it
* Import meshes from DOLFINx, GMSH, and SALOME
* Understand how the coordinate system changes the equations you solve
```

## Which page do I need?

| If you want to... | Go to | Why |
|---|---|---|
| Simulate a slab, a foil, or a depth profile | [](mesh_festim.md) | 1D only, no external tool, refine by choosing vertex spacing |
| Mesh a square, a cube, or another simple shape | [](mesh_fenics.md) | DOLFINx builds these in one call, and can scale, translate, and rotate them |
| Mesh a real CAD geometry, or one you can describe in a script | [](mesh_gmsh.md) | GMSH is scriptable, handles multi-volume geometries, and imports CAD files |
| Mesh a complex geometry using a GUI | [](mesh_salome.md) | SALOME is graphical, useful when the geometry is easier drawn than coded |
| Exploit axial or spherical symmetry | [](coordinate_systems.md) | Solving a 1D radial problem instead of a 3D one is far cheaper |

```{tip}
Reach for the simplest tool that represents your geometry. A 1D FESTIM mesh runs in seconds and is often enough to answer the question you actually have. Move to GMSH or SALOME when the geometry, rather than the physics, is what you are missing.
```