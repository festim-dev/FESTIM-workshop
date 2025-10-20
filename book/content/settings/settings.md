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

# Settings #

+++

The settings of a FESTIM simulation are defined with a `festim.Settings` object. This tutorial provides information for defining required and optional settings for users to customize their simulations.

Objectives:
* Defining tolerances and solver settings
* Setting up transient or steady-state simulations

+++

## Defining tolerances and solver settings ##

The required settings for any FESTIM simulation are the absolute and relative tolerances, while users can optionally specify the maximum number of iterations for the solver, degree order for finite element, and whether to use residual or incremental convergence criterion (for Newton solvers).

We can define the tolerances using `atol` (absolute) and `rtol` (relative):

```{code-cell} ipython3
import festim as F

settings = F.Settings(
    atol=1e10,
    rtol=1e-10
)
```

To specify the maximum number of iterations (which defaults to 30), we can use `max_iterations`:

```{code-cell} ipython3
settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    max_iterations=50
)
```

To specify the degree order of the finite element (which defaults to 1), we can use `element_degree`:

```{code-cell} ipython3
settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    element_degree=2
    )
```

To specify the convergence criterion, we can use `convergence_criterion` and strings for `residual` and `incremental`. For a residual-based convergence:

```{code-cell} ipython3
settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    convergence_criterion='residual'
)
```

## Setting up transient or steady-state simulations ##

For transient simulations, we need to define `final_time` and `stepsize`, while for steady-state problems, we simply need to set `transient` to `False`.

For example, if we have an absolute and relative tolerance of `1e10` and `1e-10`, respectively, we can define the steady-state settings as:

```{code-cell} ipython3
import festim as F

my_settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    transient=False,
)
```

For a transient simulation with a run-time of 10 seconds and stepsize of 2 seconds:

```{code-cell} ipython3
my_settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    final_time=10,
    stepsize=2
)
```

```{note}
FESTIM defaults the `transient` setting to `True`, while the stepsize and final time defaults to `None`.
```
