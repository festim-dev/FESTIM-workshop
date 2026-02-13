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

# Advanced examples

This section discusses a few advanced examples using various boundary conditions to help users understand how BCs can help model their systems.

Objective:
* Model plasma implantation using volumetric sources and approximations
* Model complex multi-species isotopic exchange surface reactions

+++

## Modeling plasma implantation

We can model plasma implantation using FESTIM's `ParticleSource` class, which is a class used to define volumetric source terms. This is helpful in modeling thermo-desorption spectra (TDS) experiments or including the effect of plasma exposure on hydrogen transport. 

Consider the following 1D plasma implantation problem, where we represent the plasma as a hydrogen source $S_{ext}$ implanted on a tungsten material that is 20 microns thick:

$$ S_{ext} = \varphi_{imp} \cdot f(x) $$
$$ L = 20 \mu \mathrm{m}$$
$$\varphi_{imp} = 10^{13} \quad \mathrm{m}^{-2}\mathrm{s}^{-1}$$

where  $\varphi_{imp}$ is the implantation flux and $f(x)$ is a Gaussian spatial distribution (distribution mean value represents implantation depth).

First, we setup a 1D mesh ranging from $ [0,L] $ and assign the subdomains and material properties for tungsten:

```{code-cell} ipython3
import festim as F
import ufl
import numpy as np

L = 20e-6
my_model = F.HydrogenTransportProblem()
vertices = np.linspace(0, L, 1000)
my_model.mesh = F.Mesh1D(vertices)

tungsten = F.Material(
    D_0=4.1e-07,  # m2/s
    E_D=0.39,  # eV
)
volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=tungsten)
left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
right_boundary = F.SurfaceSubdomain1D(id=2, x=L)

my_model.subdomains = [
    volume_subdomain,
    left_boundary,
    right_boundary,
]
```

Now, we define our `incident_flux` and `gaussian_distribution` function. Here, we define the mean implantation depth $R_p$ and distribution width to be a few nanometers long:

$$ R_p = 4 \times 10^{-9} \mathrm{m} $$
$$ \mathrm{width} = 2.5 \times 10^{-9} \mathrm{m} $$

```{code-cell} ipython3
incident_flux = 1e13
Rp = 4e-9
width = 2.5e-9

def gaussian_distribution(x, center, width):
    return (
        1
        / (width * (2 * ufl.pi) ** 0.5)
        * ufl.exp(-0.5 * ((x[0] - center) / width) ** 2)
    )
```

We can define our species and use `ParticleSource` to represent the source term, and then add it to our model:

```{code-cell} ipython3
H = F.Species("H")
my_model.species = [H]

source_term = F.ParticleSource(
    value=lambda x: incident_flux * gaussian_distribution(x, Rp, width),
    volume=volume_subdomain,
    species=H,
)

my_model.sources = [source_term]
```

Finally, we assign boundary conditions (zero concentration at $x=0$ and $x=L$) and solve our problem:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H),
    F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
]

my_model.temperature = 400
my_model.settings = F.Settings(atol=1e10, rtol=1e-10, transient=False)

profile_export = F.Profile1DExport(field=H,subdomain=volume_subdomain)
my_model.exports = [profile_export]

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

x = my_model.exports[0].x
c = my_model.exports[0].data[0]

plt.plot(x, c)
plt.xlabel("x")
plt.ylabel("c")
plt.show()
```

We see that a huge spike in concentration in the first few nanometers of tungesten, where the implantation range is focused.

+++

### Approximating plasma implantation using fixed concentration boundary conditions

If recombination is fast enough, the spike shown above can be approximated as a fixed concentration boundary condition that mainly drives diffusion across the material. Learn more about the plasma implantation approximation approach _[here](https://festim.readthedocs.io/en/fenicsx/theory.html#plasma-implantation-approximation)_.

To see how we might approximate this, let's define a maximum concentration to set on the left boundary, representing the spike from the implantation:

$$ c_m = \frac{R_p \cdot \varphi_{imp}}{D} $$

where $\varphi_{imp}$ is the implantation flux and $R_p$ is the implantation depth, both which we defined earlier, and $D$ is the material diffusivity:

```{code-cell} ipython3
D = tungsten.D_0 * np.exp(-tungsten.E_D / (F.k_B * my_model.temperature))
c_m = incident_flux * Rp / D
```

Now, we'll change our boundary conditions to represent the implantation as a fixed concentration on the left boundary, and remove the source term from our problem:

```{code-cell} ipython3
my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_boundary, value=c_m, species=H),
    F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
]

my_model.sources = []
profile_export = F.Profile1DExport(field=H,subdomain=volume_subdomain)
my_model.exports = [profile_export]

my_model.initialise()
my_model.run()
```

Let's compare the profiles from the approximation to the volumetric source results:

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

x2 = my_model.exports[0].x
c2 = my_model.exports[0].data[0]

plt.plot(x, c, label="With volumetric source")
plt.plot(x2, c2, label="Approximation")

plt.xlabel("x")
plt.ylabel("c")
plt.legend()
plt.show()
```

Using the approximation is computationally less expensive, and still provides similar diffusion profiles.

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
