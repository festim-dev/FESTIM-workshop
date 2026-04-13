---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
---

# Heat transfer #

This tutorial goes over how to define boundary conditions for heat transfer simulations.

Objectives:
* Define homogenous temperature boundary conditions
* Define heat flux boundary conditions

+++

## Imposing homogenous temperature boundary conditions ##

The temperature can be imposed on boundaries using `FixedTemperatureBC`:

```{code-cell}
from festim import FixedTemperatureBC, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
my_bc = FixedTemperatureBC(subdomain=boundary, value=10)
```

To define the temperature as space or time dependent, a function can be passed to the `value` argument, such as:

$$ \text{BC} = 10 + x^2 + t $$

```{code-cell}
from festim import FixedTemperatureBC, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
BC = lambda x, t: 10 + x[0]**2 + t

my_bc = FixedTemperatureBC(subdomain=boundary, value=BC)
```

## Imposing heat flux boundary conditions ##

+++

Heat fluxes can be imposed on boundaries using `HeatFluxBC`, which can depend on space, time, and temperature, such as:

$$ \text{BC} = 2x + 10t + T $$

```{code-cell}
from festim import HeatFluxBC, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
BC = lambda x, t, T: 2 * x[0] + 10 * t + T

my_flux_bc = HeatFluxBC(subdomain=boundary, value=BC)
```

```{note}
Read more about heat transfer settings __[here](https://festim-workshop.readthedocs.io/en/festim2/content/temperatures/temperatures_advanced)__.
```
