---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

(simple-simulation)=
# Simple simulation

In this task, we'll go through the basics of FESTIM and run a simple simulation on a 1D domain.

```{admonition} Objectives
:class: objectives

* Assemble a complete FESTIM simulation: mesh, material, species, temperature, boundary conditions, settings, exports
* Run a transient hydrogen transport problem
* Plot the concentration profile and explain its shape
```

The very first step is to import the {py:mod}`festim<festim>` package:

```{code-cell} ipython3
import festim as F

print(F.__version__)
```

We then create a {py:class}`HydrogenTransportProblem<festim.hydrogen_transport_problem.HydrogenTransportProblem>` object.

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
```

## Mesh

FESTIM simulations need a mesh, here we use {py:class}`Mesh1D<festim.mesh.Mesh1D>`.

```{code-cell} ipython3
import numpy as np

my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, num=1001))
```

```{seealso}
For more information on meshes in FESTIM, see [](meshes).
```

+++

## Materials

{py:class}`Material<festim.material.Material>` objects hold the materials properties like diffusivity.

```{code-cell} ipython3
mat = F.Material(D_0=1, E_D=0.0)

volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
boundary_left = F.SurfaceSubdomain1D(id=1, x=0)
boundary_right = F.SurfaceSubdomain1D(id=2, x=1)
my_model.subdomains = [volume_subdomain, boundary_left, boundary_right]
```

```{code-cell} ipython3
H = F.Species("H")
my_model.species = [H]
```

## Temperature

```{code-cell} ipython3
my_model.temperature = 300
```

## Boundary conditions

Our hydrogen transport problem now needs boundary conditions. We hold the concentration at 1 H/m3 on the left boundary and at 0 H/m3 on the right, which drives hydrogen through the domain from left to right.

```{admonition} Exercise: write the boundary conditions
:class: exercise

Fill in the cell below using `F.FixedConcentrationBC`. Each condition needs a `subdomain`, a `value`, and a `species`. The subdomains `boundary_left` and `boundary_right`, and the species `H`, were defined in the Materials section above.

Reading along rather than running? Expand the solution underneath.
```

```{code-cell} ipython3
:tags: [skip-execution]

my_model.boundary_conditions = [
    # TODO: fix the concentration to 1 on boundary_left
    # TODO: fix the concentration to 0 on boundary_right
]
```

```{code-cell} ipython3
:tags: [hide-cell]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=boundary_left, value=1, species=H),
    F.FixedConcentrationBC(subdomain=boundary_right, value=0, species=H),
]
```

## Settings

With {py:class}`Settings<festim.settings.Settings>` we set the main solver parameters.

```{code-cell} ipython3
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=2)
```

## Stepsize

Since we are solving a transient problem, we need to set a {py:class}``Stepsize<festim.stepsize.Stepsize>``.
Here, the value of the stepsize is fixed at 0.05.

```{code-cell} ipython3
my_model.settings.stepsize = F.Stepsize(0.05)  # s
```

## Exports

Finally, we want to be able to visualise the concentration field.

```{code-cell} ipython3
profile = F.Profile1DExport(field=H, subdomain=volume_subdomain, times=[0.05, 0.1, 0.2, 0.5, 1])

my_model.exports = [
    F.VTXSpeciesExport(
        field=H,
        filename="hydrogen_concentration.bp",
    ),
    profile,
]
```

## Run

Finally, we initialise the model and run it!

```{code-cell} ipython3
my_model.initialise()

my_model.run()
```

You should now see the file `hydrogen_concentration.bp`.

```{seealso}
Check out the [Paraview](paraview) section for visualisation!
```

+++

## Visualise

```{admonition} Exercise: predict the profile
:class: exercise

The left boundary is held at 1 H/m3 and the right at 0 H/m3. The domain is 1 m long and, because `E_D` is zero, the diffusivity is simply $D = D_0 = 1$ m2/s.

Before running the cell below, predict:

1. After 0.05 s, roughly how far into the domain has hydrogen travelled?
2. Will the 0.05 s profile be a straight line?
```

```{admonition} Solution
:class: dropdown

1. Diffusion spreads a front over a characteristic length $\sqrt{D t}$. At $t = 0.05$ s that is $\sqrt{1 \times 0.05} \approx 0.22$ m, so the front has reached only about a fifth of the way across and the far end is still essentially untouched.

2. No. It is a curved front, steepest at $x = 0$, decaying towards zero. The straight line is the *steady state*, and it is only reached once diffusion has had time to cross the whole domain: setting $\sqrt{D t} \approx L$ gives $t \approx L^2 / D = 1$ s. That is why the 1 s curve in the plot below is nearly straight, while the early ones are not.
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

x = my_model.mesh.mesh.geometry.x[:, 0]
for time, data in zip(profile.t, profile.data):
    plt.plot(x, data, label=f"{time:.2f} s")

plt.xlabel("x (m)")
plt.ylabel("Mobile concentration (H/m3)")
plt.legend()
plt.show() 
```

```{admonition} Exercise: does temperature matter here?
:class: exercise

We set `my_model.temperature = 300`. What would happen to these curves at 600 K instead? Commit to an answer before expanding.
```

```{admonition} Solution
:class: dropdown

Nothing would change at all.

FESTIM gets the diffusivity from the Arrhenius law

$$
D = D_0 \exp{(-E_D / k_B T)}
$$

and this material was defined with `E_D = 0`. The exponential is then $\exp(0) = 1$, so $D = D_0 = 1$ m2/s at *every* temperature. The temperature is doing no work in this simulation, which keeps the numbers round while you learn the API.

Temperature only bites once the activation energy is non-zero. Set `E_D=0.2` and rerun at 300 K and 600 K, and the hot case will be dramatically faster. Real materials have a non-zero `E_D`: see [](../material/material_basics.md) for how to define one, and [](../material/material_htm.md) for measured values from the literature.
```
