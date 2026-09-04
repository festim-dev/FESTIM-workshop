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

# Drift terms

```{versionadded} 2.2
`drift_terms`, `SoretTerm`, `ElectromigrationTerm` and `OutflowBC` were introduced in FESTIM 2.2.
```

By default, hydrogen moves only down its own concentration gradient. A **drift term** makes it
move because of something else: a fluid carrying it along, a temperature gradient, an electric
field. All of them add a velocity $\mathbf{v}$ to the flux

$$ J = -D \nabla c + c \, \mathbf{v} $$

and differ only in what sets $\mathbf{v}$:

| Class | Driven by | $\mathbf{v}$ |
| --- | --- | --- |
| {py:class}`festim.SoretTerm` | a temperature gradient | $-D \dfrac{Q^*}{k_B T^2} \nabla T$ |
| {py:class}`festim.ElectromigrationTerm` | an electric potential | $-\dfrac{z D}{k_B T} \nabla \varphi$ |
| {py:class}`festim.AdvectionTerm` | a moving fluid | the fluid velocity |

They are all passed to the model through the same attribute:

```python
my_model.drift_terms = [term_1, term_2, ...]
```

Every drift term is assembled in **divergence form**, $\nabla \cdot (c \mathbf{v})$, which is what
conserves the species. That choice has a consequence for boundary conditions that is worth
understanding before you use any of them, and it is covered in
[](advection.md).
