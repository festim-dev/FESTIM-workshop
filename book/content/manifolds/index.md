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

# Manifold subdomains

```{versionadded} 2.2
Codimensional (manifold) subdomains were introduced in FESTIM 2.2.
```

Some of the things hydrogen travels along are far too thin to mesh: a grain boundary, a crack, an
oxide layer, a coolant channel running along a wall. Resolving them with cells would either dominate
the mesh or force a time step nobody wants.

A **manifold subdomain** is the alternative. It is a {py:class}`festim.VolumeSubdomain` that lives on
a *line* in a 2D mesh or a *surface* in a 3D mesh — one dimension less than the mesh, hence
*codimension 1*. It carries its own transport equation, with diffusion (and advection) **along** it,
and it exchanges hydrogen with the bulk it is embedded in through a flux you write yourself.

[](manifold_basics.md) covers declaring one, coupling it to the bulk, putting boundary conditions
and reactions on it, and getting quantities out. [](pipe_wall.md) puts it to work on a coolant
channel picking up hydrogen permeating through a pipe wall.
