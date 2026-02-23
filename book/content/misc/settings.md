---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.20.0
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

## Adaptive stepsize ##

It is often useful to have an adaptive stepsize that grows or shrink based on the difficulty of the solution.

FESTIM does that by allowing users to define their own `F.Stepsize` object.

The `target_nb_iterations` sets the optimal number of Newton iterations. If more iterations are required in order to converge, this might suggest that the problem is hard to solve and a smaller stepsize is required. On the other hand, when the solver converges very quickly, this may be possible to have larger stepsizes.

The parameter `growth_factor` defines by how much the stepsize is increased, and `cutback_factor` defines by how much it is shrunk.

Let's demonstrate this with a simple example. Here we create an "empty" transient problem with no BCs, no source terms, nothing! The solution is $c=0$ everywhere. We do this so that the number of iterations required to "converge" is always below `target_nb_iterations`, and the stepsize is increased everytime.

```{code-cell} ipython3
import festim as F
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
festim_mesh = F.Mesh(fenics_mesh)

my_model = F.HydrogenTransportProblem()

material_top = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain(id=1, material=material_top)

my_model.mesh = festim_mesh
my_model.subdomains = [vol]

H = F.Species("H")
my_model.species = [H]

my_model.settings = F.Settings(atol=1e-8, rtol=1e-8, final_time=1000)

my_model.temperature = 300

my_model.exports = [F.TotalVolume(field=H, volume=vol)]
```

We define a `F.Stepsize` object with an initial value of 10 and some typical control parameters:

```{code-cell} ipython3
my_model.settings.stepsize = F.Stepsize(
    initial_value=10,
    growth_factor=1.1,  # grow by 10%
    cutback_factor=0.9,  # shrink by 10%
    target_nb_iterations=4,  # target number of iterations per time step
)
```

Let's run the adaptive time stepping model:

```{code-cell} ipython3
print("Running with adaptive time stepping...")
my_model.initialise()
my_model.run()
times_fast = my_model.exports[0].t
```

Now, let's replace the stepsize by a fixed stepsize and see how they compare:

```{code-cell} ipython3
print("Running with fixed time stepping...")
my_model.settings.stepsize = 10
my_model.initialise()
my_model.run()
times_slow = my_model.exports[0].t
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 1, sharex=True)


def plot(times, label=None, annotate=True):
    steps = np.arange(len(times))
    dt = np.diff(times)

    axs[0].plot(steps[1:], dt, label=label)
    (l,) = axs[1].plot(steps, times, label=label)
    axs[1].scatter(steps[-1], times[-1], color=l.get_color())
    if annotate:
        axs[1].annotate(
            "Final time",
            (steps[-1], times[-1]),
            textcoords="offset points",
            xytext=(-10, -5),
            ha="right",
            color=l.get_color(),
        )


plot(times_fast, label="Adaptive time stepping")
plot(times_slow, label="Fixed time stepping")

axs[0].set_ylabel(r"Step size $\Delta t$")
axs[0].set_ylim(bottom=0)
axs[1].set_ylabel("Time $t$")
axs[0].legend(loc="upper right")
plt.xlabel("Timestep")

plt.show()
```

As expected, the stepsize is growing at each time step, meaning the final time is reached in just above 20 timesteps. Whereas for the fixed time stepping, it takes 100 iterations.

+++

Stepsize can be capped by setting the parameter `max_stepsize`:

```{code-cell} ipython3
my_model.settings.stepsize = F.Stepsize(
    initial_value=10,
    growth_factor=1.1,  # grow by 10%
    cutback_factor=0.9,  # shrink by 10%
    target_nb_iterations=4,  # target number of iterations per time step
    max_stepsize=40, # maximum step size
)

my_model.initialise()
my_model.run()

capped_times = my_model.exports[0].t
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axs = plt.subplots(2, 1, sharex=True)
plot(times_fast, label="Adaptive time stepping", annotate=False)
plot(times_slow, label="Fixed time stepping", annotate=False)
plot(capped_times, label="Adaptive with max", annotate=False)

axs[0].set_ylabel(r"Step size $\Delta t$")
axs[0].set_ylim(bottom=0)
axs[1].set_ylabel("Time $t$")
axs[0].legend(loc="upper right")
plt.xlabel("Timestep")

plt.show()
```



## Custom PETSC parameters ##
