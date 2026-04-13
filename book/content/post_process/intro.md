---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
---

# Post processing #

This section explains which quantities should be exported for different post-processing objectives, and how to choose between fields and derived values.

Objectives:
- Understand the difference between fields and derived values
- Identify which quantities should be exported for common post-processing tasks

+++

## What is a field? ##

A field is a primary solution variable defined over the computational domain and represented on the mesh. Fields are directly solved for by the governing equations and correspond to vectors of degrees of freedom (DOFs) associated with a function space.

Common examples include:

$$ \text{Temperature: } T(\mathbf{x}, t) $$
$$ \text{Velocity: } \mathbf{u}(\mathbf{x}, t) $$
$$ \text{Pressure: } p(\mathbf{x}, t) $$

In FESTIM, the primary solved fields are hydrogen isotope concentration fields. At minimum, FESTIM solves for the mobile concentration:

$$ \text{Mobile concentration: } c(\mathbf{x}, t) $$

When trap populations are defined, FESTIM additionally solves for trapped concentration fields.

+++

## What is a derived value? ##

A derived value is a quantity that is computed from one or more solved fields during or after post-processing. Derived values are not primary unknowns of the governing equations and are therefore not solved for directly.

Derived values may be obtained through algebraic operations, differential operators, or spatial and temporal integrals applied to existing fields. They do not have independent degrees of freedom, although they may be projected onto a function space for visualization or storage.

Common examples include:

$$ \text{Flux: } \mathbf{J} = -D(T)\nabla c $$
$$ \text{Total inventory: } N = \int_\Omega c \, \mathrm{d}V $$

In FESTIM, quantities such as fluxes, retention, inventories, and permeation rates are derived values computed from the solved concentration fields.

+++

## Choosing which export to use ##

The type of export depends on your post-processing needs:

| Export type       | When to use |
|------------------|----------------|
| **Fields**        | Need spatially resolved data |
|                  | Compute additional quantities externally for further analysis or coupled simulations |
|                  | Reuse results in later simulations|
| **Derived values** | Need global or integrated quantities (total inventory, fluxes, etc.) |
|                  | Only scalar metrics or time histories needed |
|                  | Avoid storing full field data |

```{note}
Often both are useful: fields provide detail, derived values provide compact, directly interpretable results. See [exporting fields](exports.md) and [derived values](derived.md) for implementation details.
```

+++
