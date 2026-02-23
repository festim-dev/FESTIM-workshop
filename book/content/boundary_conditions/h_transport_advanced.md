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

# Hydrogen transport: advanced

This section discusses how to implement advanced boundary conditions (BCs) for hydrogen transport problems.

Objectives:
* Choose solubility laws (Henry's or Siervert's)
* Implement surface reaction boundary conditions
* Model isotopic exchange as a recombination flux

+++

## Understanding background for solubility laws and surface reactions

+++

### Solubility laws

A solubility law prescribes the equilibrium relationship between the hydrogen concentration in a solid and the surrounding environment. Two common solubility laws for hydrogen are **Henry’s law** and **Sieverts’ law**:

#### Henry's law

Henry’s law assumes a linear relationship between the hydrogen concentration in the solid and the partial pressure of hydrogen in the gas:

$$
c_s = k_H \, P_H
$$

where:  
- $c_s$ is the surface concentration,  
- $P_H$ is the hydrogen partial pressure in the surrounding environment,  
- $k_H$ is the Henry’s law constant (material and temperature-dependent).

```{tip}
Users interested in molten salt interactions with hydrogen will usually need to use *Henry's Law*.
```

#### Sieverts' law

Sieverts’ law applies when hydrogen dissolves in metals as atomic hydrogen. The solubility is proportional to the square root of the hydrogen partial pressure (due to diatomic hydrogen dissociating into atoms in the solid):

$$
c_s = k_S \, \sqrt{P_H}
$$

where:  
- $k_S$ is the Sieverts’ constant (which depends on temperature and the metal)

```{tip}
Users interested in liquid metal interactions with hydrogen will usually need to use **Sievert's Law**.
```

+++

### Imposing a solubility law

Users can define the surface concentration using either Sieverts’ law. For Sieverts' law of solubility, we can use `SievertsBC`:

```{code-cell} ipython3
import festim as F

boundary = F.SurfaceSubdomain(id=1)
H = F.Species(name="Hydrogen")

custom_pressure_value = lambda t: 2 + t
my_sieverts_bc = F.SievertsBC(subdomain=3, S_0=2, E_S=0.1, species=H, pressure=custom_pressure_value)
```

Similarly, for Henry's law of solubility, we can use `HenrysBC`:

```{code-cell} ipython3
pressure_value = lambda t: 5 * t
my_henrys_bc = F.HenrysBC(subdomain=3, H_0=1.5, E_H=0.2, species=H, pressure=pressure_value)
```

To use these BCs in your simulation, add them to `boundary_conditions`:

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
my_model.boundary_conditions = [my_sieverts_bc, my_henrys_bc]
```

```{note}
Using `HenrysBC` or `SievertsBC` is just a convenient way for users to define a `FixedConcentrationBC`, using the solubility law to calculate the concentration.
```

+++

---

+++

## Simple surface reaction

Surface reactions between adsorbed species and gas-phase products can be represented using the `SurfaceReactionBC` class. This boundary condition defines a reversible reaction of the form:

$$
A + B \quad \overset{K_r}{\underset{K_d}{\rightleftharpoons}} \quad C
$$

where:  
- $\mathrm{A}, \mathrm{B}$ are surface reactants  
- $\mathrm{C}$ is the product species in the gas phase  
- $\mathrm{K}_r$ and $\mathrm{K}_d$ are forward and backward reaction rates, respectively

### Mathematical formulation

Let:  
- $c_A, c_B$ be the surface concentrations of $\mathrm{A}$ and $\mathrm{B}$  
- $P_C$ be the partial pressure of $\mathrm{C}$  
- $T$ the surface temperature  
- $k_B$ the Boltzmann constant  

The forward and backward reaction rates follow Arrhenius laws:

$$
K_r = k_{r0} \exp\!\left(-\frac{E_{kr}}{k_B T}\right), \qquad
K_d = k_{d0} \exp\!\left(-\frac{E_{kd}}{k_B T}\right)
$$

The net surface reaction rate is:

$$
K = K_r \, c_A c_B - K_d \, P_C
$$

The resulting flux of species $\mathrm{A}$ into the surface is:

$$
\mathbf{J}_A \cdot \mathbf{n} = K
$$

If $\mathrm{A=B}$, the total flux becomes:

$$
\mathbf{J}_A \cdot \mathbf{n} = 2 K
$$

where $\mathbf{n}$ is the outward normal vector at the surface.

### Recombination and dissociation

If $\text{Species A = Species B}$, recombination or dissociation can also be modeled with `SurfaceReactionBC`, e.g.:

$$
\mathrm{H + H} \quad \overset{K_r}{\underset{K_d}{\rightleftharpoons}} \quad \mathrm{H_2}
$$

where 

$$
K = K_r \, c_H^2 - K_d \, P_{H_2}
$$

and corresponding flux of atomic hydrogen is:

$$
\mathbf{J}_H \cdot \mathbf{n} = - 2 K
$$

+++

### Implementing surface reaction boundary conditions

Users can impose a surface reaction on a boundary using `SurfaceReactionBC`. 

First, create a boundary to impose the BC and the reactant species:

```{code-cell} ipython3
import festim as F

boundary = F.SurfaceSubdomain(id=1)
A = F.Species("A")
B = F.Species("B")
```

Now, we add these as arguments to the `SurfaceReactionBC` class. We'll also need to assign a `gas_pressure` (corresponding to the partial pressure of the product species), and the forward/backward rate (`k_r0`, `k_d0`) and energy (`E_kr`, `E_kd`) coefficients (see above to learn more about these parameters):

```{code-cell} ipython3
my_bc = F.SurfaceReactionBC(
    reactant=[A, B],
    gas_pressure=1e5,
    k_r0=1,
    E_kr=0.1,
    k_d0=0,
    E_kd=0,
    subdomain=boundary,
)
```

Finally, add the BC to your problem's `boundary_conditions` attribute using a list:

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
my_model.boundary_conditions = [my_bc]
```

```{note}
Using `SurfaceReactionBC` is just a convenient way for users to define surface fluxes, which can be done manually (as shown below in the isotopic exchange example).
```

+++

---

+++

## Isotopic exchange

Isotopic exchange occurs when hydrogenic isotopes swap positions, for example:

$$
\mathrm{T + T} \rightleftharpoons \mathrm{T_2}
\\
\mathrm{T + H_2} \rightleftharpoons \mathrm{H} + \mathrm{HT}
$$

### Mathematical formulation

Let:  
- $c_T, c_{H_2}, c_{HT}$ be the surface concentrations of tritium, molecular hydrogen, and hydrogen-tritium  
- $K_{r0}, K_{r0}^\ast$ the pre-exponential factors  
- $E_{Kr}, E_{Kr}^\ast$ the activation energies  

If $c_{H_2} \gg c_{HT}$, the flux of tritium is approximated as:

$$
\phi_T = 
- K_{r0} \exp\!\left(-\frac{E_{Kr}}{k_B T}\right) c_T^2
- K_{r0}^\ast \exp\!\left(-\frac{E_{Kr}^\ast}{k_B T}\right) c_{H_2} c_T
$$

These fluxes can be implemented in FESTIM using `ParticleFluxBC` with user-defined expressions for each reaction term, as shown below.

+++

### Modeling isotopic exchange as a recombination flux

Isotopic exchange reactions can be modeled as user-defined expressions using `ufl`. 

Assuming a large, constant concentration of molecular hydrogen $H_2$ at a boundary, we can define a recombination flux using rate and energy coefficients:

```{code-cell} ipython3
import ufl
import festim as F

Kr_0 = 1.0
E_Kr = 0.1

def my_custom_recombination_flux(c, T):
    Kr_0_custom = 1.0
    E_Kr_custom = 0.5  # eV
    h2_conc = 1e25  # assumed constant H2 concentration in

    recombination_flux = (
        -(Kr_0 * ufl.exp(-E_Kr / (F.k_B * T))) * c**2
        - (Kr_0_custom * ufl.exp(-E_Kr_custom / (F.k_B * T))) * h2_conc * c
    )
    return recombination_flux
```

```{note}
For more complex isotopic exchanges, we can also use `SurfaceReactionBC` add other reactions. See [modeling isotopic exchange for multiple species](examples.md) to learn more.
```

+++

Let's work through a 1D example to illustrate the effect of isotopic exchange. We'll consider tritium diffusion from left to right, with a large partial pressure of $H_2$ at the right boundary (where isotopic exchange can occur).

First, we'll set up our mesh and materials:

```{code-cell} ipython3
import numpy as np
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(create_unit_square(MPI.COMM_WORLD, 300, 300))

right_subdomain = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1))
left_subdomain = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0))

material = F.Material(D_0=1e-2, E_D=0)

vol = F.VolumeSubdomain(id=5, material=material)
my_model.subdomains = [vol, left_subdomain, right_subdomain]
```

Now we'll add our species `tritium` to our model and define a custom flux to represent isotopic exchange. We use `ParticleFluxBC` and the custom flux function we defined above to define this BC on the right boundary:

```{code-cell} ipython3
tritium = F.Species("T")
my_model.species = [tritium]

my_custom_flux = F.ParticleFluxBC(
    value=my_custom_recombination_flux,
    subdomain=right_subdomain,
    species_dependent_value={"c": tritium},
    species=tritium,
)
```

For diffusion across the mesh, we also define a fixed concentration on the left surface:

```{code-cell} ipython3
my_model.boundary_conditions = [
    my_custom_flux,
    F.FixedConcentrationBC(subdomain=left_subdomain, value=1e20, species=tritium),
]
```

We'll export the flux using `SurfaceFlux` (see [exporting derived quantities to learn more](../post_process/derived.md)) and save this data to a variable `data_with_H2`:

```{code-cell} ipython3
surface_flux = F.SurfaceFlux(field=tritium, surface=right_subdomain)
```

```{code-cell} ipython3
my_model.temperature = 300
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, stepsize=50, final_time=200)
my_model.exports = [surface_flux]
my_model.initialise()
my_model.run()

data_with_H2 = (surface_flux.data)
```

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import pyvista
from dolfinx import plot

topology, cell_types, geometry = plot.vtk_mesh(tritium.post_processing_solution.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["c"] = tritium.post_processing_solution.x.array
grid.set_active_scalars("c")
```

We can compare the surface flux to the case where we have no isotopic exchange by removing the custom boundary condition and saving the results to `data_no_H2`:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_subdomain, value=1e20, species=tritium),
]

my_model.exports = [surface_flux]
my_model.initialise()
my_model.run()
data_no_H2 = (surface_flux.data)
```

```{code-cell} ipython3
:tags: [hide-input, hide-output]

topology, cell_types, geometry = plot.vtk_mesh(tritium.post_processing_solution.function_space)
grid2 = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid2.point_data["c"] = tritium.post_processing_solution.x.array
grid2.set_active_scalars("c")
```

Let's plot both `data_with_H2` and `data_no_H2` to see the flux versus time for each case:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

x = my_model.mesh.mesh.geometry.x[:, 0]
t = surface_flux.t
plt.plot(t, data_no_H2, label="No H_2")
plt.plot(t, data_with_H2, label="With H_2")

plt.xlabel("Time (s)")
plt.ylabel("Surface Flux (right)")
plt.yscale("log")
plt.legend()
plt.show()
```

We see that the flux is higher with isotopic exchange, as we'd expect. Let's also take a look at the concentration profiles (top with isotopic exchange and bottom without):

```{code-cell} ipython3
:tags: [hide-input]

pyvista.set_jupyter_backend("html")

plotter = pyvista.Plotter()

plotter.add_mesh(grid)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("concentration.png")
```

```{code-cell} ipython3
:tags: [hide-input]

pyvista.set_jupyter_backend("html")

plotter = pyvista.Plotter()
vmin, vmax = grid["c"].min(), grid2["c"].max()
plotter.add_mesh(grid2, clim=[vmin, vmax])
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("concentration.png")
```

The results without isotopic exchange show virtually no diffusion for a given inlet concentration, indicating that isotopic exchange helps enchance diffusion!
