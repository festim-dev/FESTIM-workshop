---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Backgrounc

This section discusses the math behind fixed temperature/concentration and flux boundary conditions (BCs).

+++

## Fixed concentration/temperature boundary conditions

A fixed concentration (Dirichlet) boundary condition prescribes the value of the mobile hydrogen isotope concentration at a boundary. This enforces the concentration to remain constant in time and space on the specified boundary, independent of the solution in the bulk. This also applies to a fixed temperature boundary condition.

For hydrogen transport, this boundary condition is typically used to represent surfaces in equilibrium with a gas, imposed implantation conditions, or experimentally controlled concentrations. Fixed temperatures are commonly used to model infinite heat reservoirs or sinks.

### Mathematical formulation

On a boundary $\Gamma_D$, the mobile concentration satisfies

$$
c(\mathbf{x}, t) = c_0 \quad \text{for } \mathbf{x} \in \Gamma_D,
$$

where $c_0$ is the prescribed concentration value.

This condition is enforced by directly constraining the degrees of freedom associated with the boundary.

## Flux boundary conditions

A particle flux (Neumann) boundary condition prescribes the normal flux of mobile hydrogen isotopes across a boundary. Unlike fixed concentration conditions, flux boundary conditions do not directly constrain the concentration value at the surface. Instead, they control the rate at which particles enter or leave the domain.

Flux boundary conditions are commonly used to represent implantation from a plasma, outgassing to vacuum, permeation through a surface, or symmetry boundaries where no net transport occurs.

This also applies for heat flux boundary conditions. 

### Mathematical formulation

The particle flux $\mathbf{J}$ is typically given by Fick’s law,

$$
\mathbf{J} = -D \nabla c,
$$

where $D$ is the diffusion coefficient and $c$ is the mobile concentration.

On a boundary $\Gamma_N$, a prescribed normal flux is imposed as

$$
\mathbf{J} \cdot \mathbf{n} = g(\mathbf{x}, t) \quad \text{for } \mathbf{x} \in \Gamma_N,
$$

where:
- $\mathbf{n}$ is the outward unit normal vector,
- $g(\mathbf{x}, t)$ is the imposed particle flux (positive for outward flux, negative for inward flux).

A special case is the **zero-flux (symmetry) boundary condition**, given by

$$
\mathbf{J} \cdot \mathbf{n} = 0 \quad \text{on } \Gamma,
$$

which implies no net particle transport across the boundary.

#### Weak formulation

In the weak form, flux boundary conditions appear naturally as surface integrals after integration by parts. For a test function $v$, the boundary contribution is

$$
\int_{\Gamma_N} g(\mathbf{x}, t)\, v \, \mathrm{d}\Gamma.
$$

Because of this, flux boundary conditions are often referred to as *natural boundary conditions* and do not require explicit modification of the solution space, unlike Dirichlet conditions.
