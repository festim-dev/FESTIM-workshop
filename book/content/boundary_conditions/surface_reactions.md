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

# Surface reactions #

+++

FESTIM V2.0 allows users to impose surface reactions on boundaries. Learn more about surface reactions __[here](https://festim-workshop.readthedocs.io/en/festim2/content/species_reactions/surface_reactions.html)__.

Objectives:
* Defining a simple surface reaction boundary condition 
* Recombination and dissociation
* Isotopic exhange
* Complex isotopic exchange with multple hydrogenic species

+++

## Defining a simple surface reaction boundary condition ##

Surface reactions between adsorbed species and gas-phase products can be represented using the `SurfaceReactionBC` class. This boundary condition defines a reversible reaction of the form:

$$
A + B \quad \overset{K_r}{\underset{K_d}{\rightleftharpoons}} \quad C
$$

where $\mathrm{A}$ and $\mathrm{B}$ are surface reactants and $\mathrm{C}$ is the product species in the gas phase.

### Reaction Rate Formulation ###

The forward and backward rate constants follow Arrhenius laws:

$$
K_r = k_{r0} e^{-E_{kr} / (k_B T)}, \qquad
K_d = k_{d0} e^{-E_{kd} / (k_B T)}
$$

The net surface reaction rate is given by:

$$
K = K_r c_A c_B - K_d P_C
$$

where: 
- $\mathrm{k_{r0}}, \mathrm{k_{d0}}$ are rate pre-exponential constants  
- $\mathrm{c_A}, \mathrm{c_B}$ are concentrations of reactant species $\mathrm{A}$ and $\mathrm{B}$ at the surface  
- $\mathrm{P_C}$ is the partial pressure of the product species $\mathrm{C}$  
- $\mathrm{k_B}$ is the Boltzmann constant  
- $\mathrm{T}$ is the surface temperature  

The flux of species $\mathrm{A}$ entering the surface is equal to $\mathrm{K}$ (if $\mathrm{A=B}$, the total particle flux entering the surface becomes $\mathrm{2K}$).

We can use the `SurfaceReactionBC` class to impose the surface reaction above by specifying the reactants (`reactant`), gas pressure (`gas_pressure`), forward/backward rate constants (`k_r0` and `k_d0`), and rate activation energies (`E_kr` and `E_kd`):

```{code-cell} ipython3
from festim import Species, SurfaceReactionBC, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
A = Species("A")
B = Species("B")
C = Species("C")

my_bc = SurfaceReactionBC(
    reactant=[A, B],
    gas_pressure=1e5,
    k_r0=1,
    E_kr=0.1,
    k_d0=0,
    E_kd=0,
    subdomain=boundary,
)
```

## Recombination and dissociation ##

Hydrogen recombination/dissociation can also be modelled using `SurfaceReactionBC`, such as this reaction:

$$ \mathrm{H + H} \quad \overset{K_r}{\underset{K_d}{\rightleftharpoons}} \quad \mathrm{H_2} $$

```{code-cell} ipython3
from festim import Species, SurfaceReactionBC, SurfaceSubdomain

boundary = SurfaceSubdomain(id=1)
H = Species("H")

my_bc = SurfaceReactionBC(
    reactant=[H, H],
    gas_pressure=1e5,
    k_r0=1,
    E_kr=0.1,
    k_d0=1e-5,
    E_kd=0.1,
    subdomain=boundary,
)
```

## Isotopic exchange ##

Isotopic exchange can occur where isotopes can swap positions with other isotopes, such as:

$$\mathrm{T + H_2} \rightleftharpoons \mathrm{H} + \mathrm{HT}$$

These exchange processes occur through complex mechanisms and are important in understanding the behavior of hydrogenic species:

If the $\mathrm{H_2}$ concentration is assumed much larger than $\mathrm{HT}$, the flux $\phi_T$ reduces to a first-order process in $\mathrm{T}$:

$$
\phi_T = 
-\mathrm{K_{r0}} \exp\!\left(-\frac{\mathrm{E_{Kr}}}{\mathrm{k_B T}}\right) c_T^2
-\mathrm{K_{r0}^\ast} \exp\!\left(-\frac{\mathrm{E_{Kr}^\ast}}{\mathrm{k_B T}}\right) c_{H_2} c_T
$$

Such fluxes can be implemented using `festim.ParticleFluxBC` with user-defined expressions.

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
We can also use `F.SurfaceReactionBC` to define recombination fluxes, which we do in the next section for more complex isotopic exchanges.
```

```{code-cell} ipython3
import numpy as np

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 100))

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

material = F.Material(D_0=1e-2, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]
tritium = F.Species("T")
my_model.species = [tritium]

my_custom_flux = F.ParticleFluxBC(
    value=my_custom_recombination_flux,
    subdomain=right_surf,
    species_dependent_value={"c": tritium},
    species=tritium,
)

my_model.boundary_conditions = [
    my_custom_flux,
    F.FixedConcentrationBC(subdomain=left_surf, value=1, species=tritium),
]

my_model.temperature = 300
right_flux = F.SurfaceFlux(field=tritium, surface=right_surf)

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()

data_with_H2 = (tritium.solution.x.array)
```

We can compare the surface flux to the case where we have no isotopic exchange by removing the custom boundary condition:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_surf, value=1, species=tritium),
]

my_model.initialise()
my_model.run()
data_no_H2 = (tritium.solution.x.array)
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

x = my_model.mesh.mesh.geometry.x[:, 0]
plt.plot(x, data_no_H2, label="No H_2")
plt.plot(x, data_with_H2, label="With H_2")

plt.xlabel("x (m)")
plt.ylabel("Surface Flux (right)")
plt.legend()
plt.show()
```

We see that diffusion is present with isotopic exchange!

+++

## Complex isotopic exchange with multple hydrogenic species ##

Surface reactions can involve multiple hydrogen isotopes, allowing for the modeling of complex isotope-exchange mechanisms between species. For example, in a system with both mobile hydrogen and deuteriun, various molecular recombination pathways may occur at the surface, resulting in the formation of $H_2$, $D_2$, and $HD$:

$$ \text{Reaction 1}: \mathrm{H+D} \rightarrow \mathrm{HD} \longrightarrow \phi_1 = K_{r1} c_H c_D - K_{d1}P_{HD} $$
$$ \text{Reaction 2}: \mathrm{D+D} \rightarrow \mathrm{D_2} \longrightarrow \phi_2 = 2K_{r2} c_D^2 - K_{d2}P_{D2} $$
$$ \text{Reaction 3}: \mathrm{D+H_2} \rightarrow \mathrm{HD + H} \longrightarrow \phi_3 = K_{r3} c_H c_D - K_{d3}P_{HD} $$
$$ \text{Reaction 4}: \mathrm{H+H} \rightarrow \mathrm{H_2} \longrightarrow \phi_4 = 2K_{r4} c_H^2 - K_{d4}P_{H2} $$

Now consider the case where deuterium diffuses from left to right and reacts with background 
$\mathrm{H_2}$, while $\mathrm{P_{HD}}$ and $\mathrm{P_{D_2}}$ are negligible at the surface. 
Formation of $\mathrm{H}$ at the right boundary induces back-diffusion toward the left, 
even though none existed initially. 

The boundary conditions for this scenario are:
$$
c_D(x=0) = 1, \qquad c_H(x=0) = 0, \qquad P_{H2}(x=1) = \text{1000 Pa}
$$

First, let's define a 1D mesh ranging from $\mathrm{x=[0,1]}$:

```{code-cell} ipython3
import numpy as np
import festim as F

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 100))

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

material = F.Material(D_0=1e-2, E_D=0)
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]
```

Now, we define our species at recombination reactions using `SurfaceReactionBC`:

```{code-cell} ipython3
H = F.Species("H")
D = F.Species("D")
my_model.species = [H, D]

H2 = F.SurfaceReactionBC(
    reactant=[H, H],
    gas_pressure=100000,
    k_r0=1,
    E_kr=0.1,
    k_d0=1e-5,
    E_kd=0.1,
    subdomain=right_surf,
)

HD = F.SurfaceReactionBC(
    reactant=[H, D],
    gas_pressure=0,
    k_r0=1,
    E_kr=0.1,
    k_d0=1e-5,
    E_kd=0.1,
    subdomain=right_surf,
)

D2 = F.SurfaceReactionBC(
    reactant=[D, D],
    gas_pressure=0,
    k_r0=1,
    E_kr=0.1,
    k_d0=1e-5,
    E_kd=0.1,
    subdomain=right_surf,
)
```

Finally, we add our boundary conditions and solve the steady-state problem:

```{code-cell} ipython3
my_model.boundary_conditions = [
    H2,
    D2,
    HD,
    F.FixedConcentrationBC(subdomain=left_surf, value=1, species=D),
    F.FixedConcentrationBC(subdomain=left_surf, value=0, species=H),
]

my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

def plot_profile(species, **kwargs):
    c = species.post_processing_solution.x.array[:]
    x = species.post_processing_solution.function_space.mesh.geometry.x[:,0]
    return plt.plot(x, c, **kwargs)

for species in my_model.species:
    plot_profile(species, label=species.name)

plt.xlabel('Position')
plt.ylabel('Concentration')
plt.legend()
plt.show()
```

We see that the background $\mathrm{H_2}$ reacts with the $\mathrm{D}$, removing the total amount of $\mathrm{D}$ from the surface. Conversely, the $\mathrm{H}$ diffuses from the surface towards the left.
