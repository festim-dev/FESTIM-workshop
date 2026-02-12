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
* Learn mathematical formulation for solubility laws and surface reactions
* Choose solubility laws (Henry's or Siervert's)
* Implement surface reaction boundary conditions
* Model isotopic exchange as a recombination flux

+++

## Learn mathematical formulation for solubility laws and surface reactions

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

#### Weak formulation

In the weak form, solubility is imposed as a **Robin-type boundary condition**:

$$
\int_{\Gamma_s} k_s \left(c - c_s \right) v \, \mathrm{d}\Gamma
$$

where $k_s$ is a surface reaction rate constant controlling the exchange between bulk and surface concentration. This allows modeling of adsorption/desorption and surface-limited transport at material boundaries.

---

+++

### Simple surface reaction

Surface reactions between adsorbed species and gas-phase products can be represented using the `SurfaceReactionBC` class. This boundary condition defines a reversible reaction of the form:

$$
A + B \quad \overset{K_r}{\underset{K_d}{\rightleftharpoons}} \quad C
$$

where:  
- $\mathrm{A}, \mathrm{B}$ are surface reactants  
- $\mathrm{C}$ is the product species in the gas phase  
- $\mathrm{K}_r$ and $\mathrm{K}_d$ are forward and backward reaction rates, respectively

#### Mathematical formulation

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

#### Weak form contribution

In the weak formulation, the surface reaction contributes as a Robin-type term:

$$
\int_{\Gamma_s} \mathbf{J}_A \cdot \mathbf{n} \, v \, d\Gamma = 
\int_{\Gamma_s} K \, v \, d\Gamma
$$

where $v$ is a test function.

---

+++

### Recombination and dissociation

Hydrogen recombination or dissociation can be modeled with `SurfaceReactionBC`, e.g.:

$$
\mathrm{H + H} \quad \overset{K_r}{\underset{K_d}{\rightleftharpoons}} \quad \mathrm{H_2}
$$

#### Mathematical formulation

Let:  
- $c_H$ be the surface concentration of atomic hydrogen  
- $P_{H_2}$ the partial pressure of molecular hydrogen  

The net reaction rate is:

$$
K = K_r \, c_H^2 - K_d \, P_{H_2}
$$

The corresponding flux of atomic hydrogen is:

$$
\mathbf{J}_H \cdot \mathbf{n} = - 2 K
$$

#### Weak form contribution

The weak form contribution of recombination flux is:

$$
\int_{\Gamma_s} \mathbf{J}_H \cdot \mathbf{n} \, v \, d\Gamma = 
- \int_{\Gamma_s} 2 K \, v \, d\Gamma
$$

---

+++

### Isotopic exchange

Isotopic exchange occurs when hydrogenic isotopes swap positions, for example:

$$
\mathrm{T + H_2} \rightleftharpoons \mathrm{H} + \mathrm{HT}
$$

#### Mathematical formulation

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

#### Weak form contribution

The weak form contribution for tritium flux is:

$$
\int_{\Gamma_s} \phi_T \, v \, d\Gamma =
- \int_{\Gamma_s} \Bigg[
K_{r0} \exp\!\left(-\frac{E_{Kr}}{k_B T}\right) c_T^2
+ K_{r0}^\ast \exp\!\left(-\frac{E_{Kr}^\ast}{k_B T}\right) c_{H_2} c_T
\Bigg] v \, d\Gamma
$$

These fluxes can be implemented in FESTIM using `ParticleFluxBC` with user-defined expressions for each reaction term (see [here](examples.ipynb) to learn more).

+++

## Implementing simple surface reaction boundary conditions

Users can impose a surface reaction at boundary using `SurfaceReactionBC`. 

First, create a boundary to impose the BC and the reactant species:

```{code-cell} ipython3
import festim as F

boundary = F.SurfaceSubdomain(id=1)
A = F.Species("A")
B = F.Species("B")
```

Now, we add these as arguments to the `SurfaceReactionBC` class. We'll also need to assign a `gas_pressure` (corresponding to the partial pressure of the product species), and the corresponding rate (`k_r0`, `k_d0`) and energy (`E_kr`, `E_dr`) coefficients (see above to learn more about these parameters):

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

### Modeling recombination and dissociation

Recombination and dissociation can also be modeled using `SurfaceReactionBC`, where the forward and backward rates of this reaction correspond to recombination and dissociation, respectively. 

To model the reaction:

$$ \mathrm{H} + \mathrm{H} \rightleftharpoons \mathrm{H_2}$$

where $ \text{Species A} = \text{Species B} = \text{H} $, assign your `reactants` list accordingly:

```{code-cell} ipython3
H = F.Species("H")

my_recombination_bc = F.SurfaceReactionBC(
    reactant=[A, A],
    gas_pressure=1e5,
    k_r0=1,
    E_kr=0.1,
    k_d0=1e-5,
    E_kd=0.1,
    subdomain=boundary,
)
```

## Modeling isotopic exchange as a recombination flux

Isotopic exchange reactions can be modeled as user-defined expressions using `ufl`. 

Assuuming a large, constant concentration of molecular hydrogen $H_2$ at a boundary, we can define a recombination flux using rate and energy coefficients:

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
For more complex isotopic exchanges, we can also use `SurfaceReactionBC` to define recombination fluxes. See [NOTE] to learn more.
```
