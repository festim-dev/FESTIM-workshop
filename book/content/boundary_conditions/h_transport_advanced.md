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

```{math}
:label: t2_recomb
\mathrm{T + T} \rightleftharpoons \mathrm{T_2}
```

```{math}
:label: isotopic_exchange
\mathrm{T + H_2} \rightleftharpoons \mathrm{H} + \mathrm{HT}
```

### Mathematical formulation

Let:  
- $c_T$ be the surface concentrations of tritium
- $P_\mathrm{H_2}, P_\mathrm{HT}$ the partial pressures of $\mathrm{H_2}$ and $\mathrm{HT}$
- $K_{r}$ the rate coefficient of reaction {eq}`t2_recomb`
- $K_{r}^\ast$ the rate coefficient of reaction {eq}`isotopic_exchange`

$$
K_{r} = K_{r0} \exp\!\left(-\frac{E_{Kr}}{k_B T}\right)
$$

$$
K_{r}^\ast = K_{r0}^\ast \exp\!\left(-\frac{E_{Kr}^\ast}{k_B T}\right)
$$

If $c_{H_2} \gg c_{HT}$, the flux of tritium is approximated as:

$$
\phi_T = \phi_\mathrm{recombination} + \phi_\mathrm{exchange}
$$
with
$$
\phi_\mathrm{recombination} = - K_{r} c_T^2 \\
\phi_\mathrm{exchange} = - K_{r}^\ast P_{H_2} c_T
$$

These fluxes can be implemented in FESTIM using `ParticleFluxBC` with user-defined expressions for each reaction term, as shown below.

+++

```{note}
$K_{r0}^\ast$ and $K_{r0}$ have different dimensions since $P_\mathrm{H_2}$ is a pressure.
```

```{note}
This example is not tracking the dissolution and transport of protium in the metal. And assumes that the partial presure of $\mathrm{H_2}$ is constant.
```

+++

```{note}
For more complex isotopic exchanges, we can also use `SurfaceReactionBC` add other reactions. See [modeling isotopic exchange for multiple species](examples.md) to learn more.
```

+++

### Implementation

+++

Let's work through a 1D example to illustrate the effect of isotopic exchange. We'll consider tritium diffusion from left to right, with a large partial pressure of $H_2$ at the right boundary (where isotopic exchange can occur).

First, we'll set up our mesh and materials:

```{code-cell} ipython3
import numpy as np

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 100))

right_subdomain = F.SurfaceSubdomain1D(id=2, x=1)
left_subdomain = F.SurfaceSubdomain1D(id=3, x=0)

material = F.Material(D_0=1e-2, E_D=0)

vol = F.VolumeSubdomain1D(id=5, material=material, borders=[0, 1])
my_model.subdomains = [vol, left_subdomain, right_subdomain]
```

Isotopic exchange reactions can be modeled as user-defined expressions using `ufl`. 

```{code-cell} ipython3
import ufl
import festim as F

Kr_0_custom = 1
E_Kr_custom = 0.0  # eV
P_H2 = 10  # assumed constant H2 concentration in

def isotopic_exchange_flux_func(c, T):
    return - (Kr_0_custom * ufl.exp(-E_Kr_custom / (F.k_B * T))) * P_H2 * c
```

Now we'll add our species `tritium` to our model and define a custom flux to represent isotopic exchange. We use `ParticleFluxBC` and the custom flux function we defined above to define this BC on the right boundary:

```{code-cell} ipython3
tritium = F.Species("T")
my_model.species = [tritium]

isotopic_exchange_flux = F.ParticleFluxBC(
    value=isotopic_exchange_flux_func,
    subdomain=right_subdomain,
    species_dependent_value={"c": tritium},
    species=tritium,
)
```

```{code-cell} ipython3
t2_recomb = F.SurfaceReactionBC(
    reactant=[tritium, tritium],
    gas_pressure=0,
    k_r0=1e-22,
    E_kr=0,
    k_d0=0,
    E_kd=0,
    subdomain=right_subdomain,
)
```

For diffusion across the mesh, we also define a fixed concentration on the left surface:

```{code-cell} ipython3
my_model.boundary_conditions = [
    isotopic_exchange_flux,
    t2_recomb,
    F.FixedConcentrationBC(subdomain=left_subdomain, value=1e20, species=tritium),
]
```

We'll export the flux using `SurfaceFlux` (see [exporting derived quantities to learn more](../post_process/derived.md)) and save this data to a variable `data_with_H2`. We also create a `Profile1DExport` to be able to plot the concentration profile later:

```{code-cell} ipython3
surface_flux = F.SurfaceFlux(field=tritium, surface=right_subdomain)

profile = F.Profile1DExport(field=tritium, subdomain=vol)
```

```{code-cell} ipython3
my_model.temperature = 300
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, stepsize=1, final_time=120)

my_model.exports = [surface_flux, profile]
my_model.initialise()
my_model.run()

data_with_H2 = surface_flux.data.copy()
```

Let's plot the evolution of the tritium concentration profile:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

import matplotlib.animation as animation
from IPython.display import HTML

# Turn off interactive plotting to prevent static display
plt.ioff()

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set up the plot limits and labels
ax.set_ylabel("Tritium concentration")
ax.set_xlabel("Position")

# Initialize empty line objects for the animation
line_current, = ax.plot([], [], color="black", linewidth=2, zorder=1000)
lines_previous = []

def animate(frame):
    # Clear previous iteration lines
    for line in lines_previous:
        line.remove()
    lines_previous.clear()
    
    # Plot all previous iterations in light grey
    for i in range(frame):
        line_prev, = ax.plot(profile.x, profile.data[i], color="lightgrey", linewidth=0.8, alpha=0.6)
        lines_previous.append(line_prev)
    
    # Plot current iteration in black
    line_current.set_data(profile.x, profile.data[frame])
    
    ax.set_title(f"Time: {profile.t[frame]:.2f}")

    return [line_current] + lines_previous

# Create animation
anim = animation.FuncAnimation(
    fig, animate, frames=len(profile.data), interval=100, blit=False, repeat=True
)

# Close the figure to prevent static display
plt.close(fig)

# Display only the animation
HTML(anim.to_jshtml())
```

We can compare the surface flux to the case where we have no isotopic exchange by removing the isotopic exchange boundary condition and saving the results to `data_no_H2`:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_subdomain, value=1e20, species=tritium),
    t2_recomb,
]

profile = F.Profile1DExport(field=tritium, subdomain=vol)
my_model.exports = [surface_flux, profile]
my_model.initialise()
my_model.run()
data_no_H2 = surface_flux.data.copy()
```

Without isotopic exchange the gradient is much lower, indicating that isotopic exchange helps enhance surface outgassing of tritium!

```{code-cell} ipython3
:tags: [hide-input]

# Create animation
anim = animation.FuncAnimation(
    fig, animate, frames=len(profile.data), interval=100, blit=False, repeat=True
)

# Close the figure to prevent static display
plt.close(fig)

# Display only the animation
HTML(anim.to_jshtml())
```

Let's plot both `data_with_H2` and `data_no_H2` to see the temporal evolution of the surface flux for each case:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

x = my_model.mesh.mesh.geometry.x[:, 0]
t = surface_flux.t
plt.plot(t, data_no_H2, label="No H_2")
plt.plot(t, data_with_H2,label="With H_2")

plt.xlabel("Time")
plt.ylabel("Surface Flux (right)")
plt.legend(reverse=True)
plt.show()
```

We see that the flux is higher with isotopic exchange, as we'd expect.
