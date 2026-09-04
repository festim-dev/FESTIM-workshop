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

# Reactions with arbitrary rates

```{versionadded} 2.2
`GenericReaction`, `DecayReaction` and `ReactionBase` were introduced in FESTIM 2.2, and
`Reaction` was renamed to `ArrheniusReaction`.
```

[](reactions.ipynb) covers trapping and detrapping, where both rate coefficients follow Arrhenius
laws. That is the common case, but not the only one. This page covers the rest of the reaction
family: rates that are not Arrhenius, rates that depend on other concentrations, net rates that are
not mass-action at all, and radioactive decay.

Objectives:
* Knowing which reaction class to reach for
* Writing a rate coefficient as a function of temperature or of another species
* Modelling tritium decay

+++

## The reaction family

Reactions form a small hierarchy, each class narrowing the one above it:

| Class | Net rate $R$ | Use it for |
| --- | --- | --- |
| {py:class}`festim.ReactionBase` | whatever you write | a rate that is not mass-action |
| {py:class}`festim.GenericReaction` | $k_1 \prod_i c_i - k_2 \prod_j c_j$ | mass action with rate coefficients you supply |
| {py:class}`festim.ArrheniusReaction` | as above, with $k = k_0 e^{-E_k / k_B T}$ | trapping, detrapping, the usual case |
| {py:class}`festim.DecayReaction` | $\lambda c$, with $\lambda = \ln 2 / t_{1/2}$ | radioactive decay |

None of them enters the formulation directly. Each is expanded into volumetric
{py:class}`festim.ParticleSource` objects — a sink $-R$ for every appearance in `reactant`, a source
$+R$ for every appearance in `product` — so a species listed twice as a reactant is consumed at rate
$2R$.

```{warning}
`F.Reaction` is now a deprecated alias for {py:class}`festim.ArrheniusReaction`. It still works and
takes the same arguments, but emits a `DeprecationWarning`. Replace

    F.Reaction(reactant=..., product=..., k_0=..., E_k=..., volume=...)

with

    F.ArrheniusReaction(reactant=..., product=..., k_0=..., E_k=..., volume=...)
```

+++

## Rate coefficients that are not Arrhenius

{py:class}`festim.GenericReaction` keeps the mass-action form but lets you write the two rate
coefficients yourself:

$$ R = k_1 \prod_i c_i^{\text{reactant}} - k_2 \prod_j c_j^{\text{product}} $$

`forward_rate` and `backward_rate` are {py:class}`festim.Value` objects, so each can be a float, a
ufl expression, or a callable of the temperature (argument `T`), the spatial coordinate (`x`) or the
time (`t`). Leaving `backward_rate` as `None` makes the reaction irreversible.

Here is a reversible conversion $A \rightleftharpoons B$ whose forward coefficient grows linearly
with temperature rather than exponentially — a fitted rate, say, rather than a thermally activated
one:

```{code-cell} ipython3
import numpy as np
import festim as F

TEMPERATURE = 500.0


def forward_rate(T):
    return 1e-3 * T  # 0.5 /s at 500 K


BACKWARD_RATE = 2.0  # /s
```

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1e-3, 20))

material = F.Material(D_0=1e-7, E_D=0.0)
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1e-3], material=material)
my_model.subdomains = [vol]

A = F.Species("A")
B = F.Species("B")
my_model.species = [A, B]

my_model.temperature = TEMPERATURE
my_model.initial_conditions = [
    F.InitialConcentration(value=1.0, species=A, volume=vol),
    F.InitialConcentration(value=0.0, species=B, volume=vol),
]

my_model.reactions = [
    F.GenericReaction(
        volume=vol,
        reactant=A,
        product=B,
        forward_rate=forward_rate,
        backward_rate=BACKWARD_RATE,
    )
]

inventory_A = F.AverageVolume(field=A, volume=vol)
inventory_B = F.AverageVolume(field=B, volume=vol)
my_model.exports = [inventory_A, inventory_B]

my_model.settings = F.Settings(atol=1e-15, rtol=1e-12, final_time=3.0)
my_model.settings.stepsize = F.Stepsize(0.05)

my_model.initialise()
my_model.run()
```

Nothing varies in space, so this is really a pair of coupled ODEs whose solution is known:
$c_A$ relaxes towards $k_2 / (k_1 + k_2)$ with a time constant $1 / (k_1 + k_2)$.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

k_1, k_2 = forward_rate(TEMPERATURE), BACKWARD_RATE
A_eq = k_2 / (k_1 + k_2)

t = np.array(inventory_A.t)
exact_A = A_eq + (1 - A_eq) * np.exp(-(k_1 + k_2) * t)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, inventory_A.data, label="A")
ax.plot(t, inventory_B.data, label="B")
ax.plot(t, exact_A, "k--", lw=1, label="A, analytical")
ax.axhline(A_eq, color="grey", lw=0.8, ls=":")
ax.set_xlabel("t (s)")
ax.set_ylabel("concentration")
ax.legend()
plt.show()
```

```{code-cell} ipython3
error = np.max(np.abs(np.array(inventory_A.data) - exact_A))
print(f"max deviation from the analytical solution: {error:.2e}")
assert error < 1e-2
```

```{tip}
`ArrheniusReaction` is just this class with the two coefficients built from `k_0`, `E_k`, `p_0` and
`E_p`. If your rate *is* an Arrhenius law, keep using it — it is clearer at a glance.
```

+++

## Rates that depend on another concentration

A rate coefficient can also depend on the concentration of a species. Declare the mapping from the
callable's argument names to {py:class}`festim.Species` objects through `arg_to_species`:

```{code-cell} ipython3
C = F.Species("C")

inhibited = F.GenericReaction(
    volume=vol,
    reactant=A,
    product=B,
    forward_rate=lambda c_C: 1.0 / (1.0 + c_C),  # C poisons the reaction
    arg_to_species={"c_C": C},
)
print(inhibited)
```

Every argument of a rate coefficient other than the reserved `t`, `x` and `T` has to appear as a key
in `arg_to_species`, and every value has to be a `Species`. FESTIM raises if a coefficient depends on
something the mapping does not name, and warns about keys no coefficient uses.

```{note}
The mapping can instead be attached to a rate passed as a {py:class}`festim.Value`, through its
`species_dependent_value` — the same mechanism used by `CustomFieldExport` in
[](../post_process/exports.md). The two ways are mutually exclusive: giving a mapping both places
raises.
```

+++

## Net rates that are not mass action

When the rate is not a product of concentrations at all, drop down to
{py:class}`festim.ReactionBase` and write $R$ directly. Everything else — how the reaction is
expanded into sources, the stoichiometry rules, `arg_to_species` — behaves the same.

```{code-cell} ipython3
exchange = F.ReactionBase(
    reaction_rate=lambda c_A, c_B: 2.0 * (c_A - c_B),
    volume=vol,
    reactant=A,
    product=B,
    arg_to_species={"c_A": A, "c_B": B},
)
print(exchange)
```

`R` is whatever the callable returns, so rates like $R = k(c_1 - c_2)$ — linear in a *difference*
rather than in a product — are expressible, which mass action cannot do.

+++

## Radioactive decay

{py:class}`festim.DecayReaction` is a first-order decay, consuming one reactant at
$R = \lambda c$ with $\lambda = \ln 2 / t_{1/2}$. Any products are optional: give them if you want to
track what the decay turns into, leave them out if you do not.

The obvious application is tritium, which decays to helium-3 with a half-life of about 12.3 years:

```{code-cell} ipython3
HALF_LIFE = 3.888e8  # s, about 12.32 years
L = 1e-3  # m
```

```{important}
`half_life` is expressed in the **simulation's time unit** — seconds, in every model in this book. A
decay is first order in the decaying species, so exactly one reactant is allowed; a list of more
than one is rejected.
```

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, L, 20))

vol = F.VolumeSubdomain1D(
    id=1, borders=[0, L], material=F.Material(D_0=1e-7, E_D=0.2)
)
my_model.subdomains = [vol]

T = F.Species("T")  # tritium
He = F.Species("He", mobile=False)  # helium-3, which does not diffuse
my_model.species = [T, He]

my_model.temperature = 500
my_model.initial_conditions = [F.InitialConcentration(value=1.0, species=T, volume=vol)]

my_model.reactions = [
    F.DecayReaction(reactant=T, half_life=HALF_LIFE, volume=vol, product=He)
]

inventory_T = F.TotalVolume(field=T, volume=vol)
inventory_He = F.TotalVolume(field=He, volume=vol)
my_model.exports = [inventory_T, inventory_He]

my_model.settings = F.Settings(atol=1e-15, rtol=1e-12, final_time=3 * HALF_LIFE)
my_model.settings.stepsize = F.Stepsize(HALF_LIFE / 40)

my_model.initialise()
my_model.run()
```

The slab is closed, so the tritium inventory should follow $N_0 e^{-\lambda t}$ and every tritium
atom lost should turn up as helium:

```{code-cell} ipython3
:tags: [hide-input]

t = np.array(inventory_T.t)
N_0 = 1.0 * L  # uniform concentration of 1 over a slab of length L
exact_T = N_0 * np.exp(-np.log(2) * t / HALF_LIFE)

fig, ax = plt.subplots(figsize=(6, 4))
years = t / (365.25 * 24 * 3600)
ax.plot(years, np.array(inventory_T.data) / N_0, label="T")
ax.plot(years, np.array(inventory_He.data) / N_0, label="He")
ax.plot(years, exact_T / N_0, "k--", lw=1, label=r"$e^{-\lambda t}$")
ax.set_xlabel("t (years)")
ax.set_ylabel("inventory (normalised)")
ax.legend()
plt.show()
```

```{code-cell} ipython3
total = (np.array(inventory_T.data) + np.array(inventory_He.data)) / N_0
print(f"T + He, worst deviation from 1: {np.max(np.abs(total - 1)):.2e}")
assert np.max(np.abs(total - 1)) < 1e-8

decay_error = np.max(np.abs(np.array(inventory_T.data) - exact_T) / exact_T)
print(f"tritium inventory, max relative error: {decay_error:.2e}")
assert decay_error < 3e-2
```

Nothing is lost: the sum of the two inventories is constant to round-off, because the sink on the
tritium and the source on the helium are the same $R$.

The couple of percent on the tritium curve is **time discretisation**, not the reaction. FESTIM steps
in time with backward Euler, which is first order, so a step of a fortieth of a half-life leaves an
error of order $\lambda \Delta t$. Halving the step halves it — see [](../misc/stepsize.md) for how
to choose one.

```{tip}
Leave `product` out if the decay product does not matter to you:

    F.DecayReaction(reactant=T, half_life=HALF_LIFE, volume=vol)

The tritium is then simply removed. Note also that helium is declared `mobile=False` above: helium
produced in a metal lattice does not diffuse on the timescales tritium does.
```
