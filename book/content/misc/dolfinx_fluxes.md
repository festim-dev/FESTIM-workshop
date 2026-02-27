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

# Boundary condition: Particle fluxes

+++

Setting up particle fluxes as boundary conditions (Neumann or Robin type) is slightly different than enforcing the concentration values at a boundary (Dirichlet type) as fluxes are added to the variational formulation directly.

Let's illustrate it with an example:

```{code-cell} ipython3
import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
import basix
from scifem import assemble_scalar
```

```{code-cell} ipython3
nx = ny = 20

domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
```

```{code-cell} ipython3
cg_element = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)

mixed_element = basix.ufl.mixed_element([cg_element])

V = fem.functionspace(domain, mixed_element)

u = fem.Function(V)

(cm,) = ufl.split(u)
(v_cm,) = ufl.TestFunctions(V)
```

```{note}
"Why make a `mixed_element` of only one element?" you may ask.

This is how it's done inside FESTIM, the main reason is it makes it possible to always process objects the same way instead of having the code full of checks to see if we have a mixed-element or not.
```

+++

We first need to create "mesh markers". We do this by using `locate_entities_boundary` with locators:

```{code-cell} ipython3
def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], 1)

fdim = domain.topology.dim - 1
entities_left = dolfinx.mesh.locate_entities_boundary(domain, fdim, left)
entities_right = dolfinx.mesh.locate_entities_boundary(domain, fdim, right)

all_entities = np.concatenate([entities_left, entities_right])
```

```{code-cell} ipython3
values_left = np.full_like(entities_left, 1.0)
values_right = np.full_like(entities_right, 2.0)

all_values = np.concatenate([values_left, values_right])
```

```{code-cell} ipython3
print(all_entities)
print(all_values)
```

```{code-cell} ipython3
facet_meshtags = dolfinx.mesh.meshtags(domain, fdim, entities=all_entities, values=all_values)
```

```{code-cell} ipython3
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_meshtags)
```

```{code-cell} ipython3
D = 1.0 # diffusion coefficient
K_r = 0.5 # recombination rate
K_d = 1 # dissociation rate

# Partial pressure
V_cg = dolfinx.fem.functionspace(domain, ("CG", 1))
P = dolfinx.fem.Function(V_cg)
P.interpolate(lambda x: 100.0 + 50.0*x[1])

F_diff = D*ufl.dot(ufl.grad(cm), ufl.grad(v_cm)) * ufl.dx

F_flux_right = K_r * cm**2 * v_cm * ds(2) - K_d * P * v_cm * ds(2)

F = F_diff + F_flux_right
```

```{code-cell} ipython3
dofs_left = fem.locate_dofs_geometrical((V.sub(0), V.sub(0).collapse()[0]), left)

bc_left = fem.dirichletbc(dolfinx.fem.Constant(domain, 0.0), dofs_left[0], V.sub(0))
```

```{code-cell} ipython3
# taken from https://github.com/FEniCS/dolfinx/blob/5fcb988c5b0f46b8f9183bc844d8f533a2130d6a/python/demo/demo_cahn-hilliard.py#L279C1-L286C28
use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
sys = PETSc.Sys()  # type: ignore
if sys.hasExternalPackage("mumps") and not use_superlu:
    linear_solver = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    linear_solver = "superlu_dist"
else:
    linear_solver = "petsc"

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(dolfinx.default_real_type).eps) * 1e-2,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-10,
    "snes_max_it": 100,
    "snes_divergence_tolerance": "PETSC_UNLIMITED",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": linear_solver,
    "snes_monitor": None,
}

problem = NonlinearProblem(
    F,
    u,
    bcs=[bc_left],
    petsc_options=petsc_options,
    petsc_options_prefix="fluxes",
)


problem.solve()

converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()

assert converged > 0, f"Solver did not converge, got {converged}."
print(
    f"Solver converged after {num_iter} iterations with converged reason {converged}."
)
```

```{code-cell} ipython3
import pyvista
from dolfinx import plot

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(u.sub(0).function_space.collapse()[0])

u_plotter = pyvista.Plotter()
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["cm"] = u.x.array.real
u_grid.set_active_scalars("cm")
u_plotter.add_mesh(u_grid, show_edges=False)

u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
```
