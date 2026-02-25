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

# Build FESTIM from scratch #

+++

Objectives:
* Understand the FESTIM foundations
* Be better armed for contributing to FESTIM

```{code-cell} ipython3
import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
from basix.ufl import element, mixed_element
```

```{code-cell} ipython3
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
```

```{code-cell} ipython3
# we create a mixed element with two components, both continuous galerkin degree 1
cg_element = element("Lagrange", domain.basix_cell(), degree=1)

mixed_element = mixed_element([cg_element, cg_element])

# then we make a functionspace from the mixed element
V = fem.functionspace(domain, mixed_element)

# we create a "main" function u which is a vector of the two components
u = fem.Function(V)

# to use the components in variational forms, we use ufl.split
# the first will be the mobile concentration cm, the second the trapped concentration ct
cm, ct = ufl.split(u)

# we create test functions for both components
v_cm, v_ct = ufl.TestFunctions(V)
```

```{code-cell} ipython3
# Boundary conditions:

def boundary_left(x):
    return np.isclose(x[0], 0)


def boundary_right(x):
    return np.isclose(x[0], 1)


def boundary_top(x):
    return np.isclose(x[1], 1)


V0, submap = V.sub(0).collapse()

# the trick here was to pass both the subspace and the collapsed space to locate_dofs_geometrical
# in FESTIM we don't need this since we use meshtags for everything
# https://fenicsproject.discourse.group/t/dolfinx-dirichlet-bcs-for-mixed-function-spaces/7844/2

dofs_right = fem.locate_dofs_geometrical((V.sub(0), V0), boundary_right)
dofs_left = fem.locate_dofs_geometrical((V.sub(0), V0), boundary_left)
dofs_top = fem.locate_dofs_geometrical((V.sub(0), V0), boundary_top)

# FIXME the following doesn't work for some reason but I opened an issue on the FEniCS discourse
# https://fenicsproject.discourse.group/t/mixedelement-functionspace-dirichletbc-with-function-as-value/19438

# uD = fem.Function(V0)
# uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
# bc_left = fem.dirichletbc(uD, dofs_left[0])

bc_left = fem.dirichletbc(fem.Constant(domain, 2.0), dofs_left[0], V.sub(0))
bc_right = fem.dirichletbc(fem.Constant(domain, 1.0), dofs_right[0], V.sub(0))
bc_top = fem.dirichletbc(fem.Constant(domain, 3.0), dofs_top[0], V.sub(0))
```

```{code-cell} ipython3
# Problem parameters
k = 0.01  # trapping rate
p = 0.1  # detrapping rate
n = 0.1  # total trapping sites

trapping = k * cm * (n - ct)
detrapping = p * ct

# NOTE everything is bundled in one variational form F
# the difference between the different equations is made with the test functions v_cm and v_ct
F_mobile = (
    ufl.dot(ufl.grad(cm), ufl.grad(v_cm)) * ufl.dx
    - trapping * v_cm * ufl.dx
    + detrapping * v_cm * ufl.dx
)
F_trapped = +trapping * v_ct * ufl.dx - detrapping * v_ct * ufl.dx

F = F_mobile + F_trapped
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
}

problem = NonlinearProblem(
    F,
    u,
    bcs=[bc_left, bc_right, bc_top],
    petsc_options=petsc_options,
    petsc_options_prefix="Poisson",
)
problem.solve()
converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()
assert converged > 0, f"Solver did not converge, got {converged}."
print(
    f"Solver converged after {num_iter} iterations with converged reason {converged}."
)

## Post-processing


# we first split the main solution u into its components with .split()
cm_post, ct_post = u.split()  # NOTE this is different from ufl.split(u)

# for postprocessing, it's easier to work with collapsed functions
cm_post = cm_post.collapse()
ct_post = ct_post.collapse()
```

```{code-cell} ipython3
# Visualization

import pyvista
from dolfinx import plot

domain.topology.create_connectivity(tdim, tdim)
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(cm_post.function_space)


# plot cm
u_plotter = pyvista.Plotter()
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["cm"] = cm_post.x.array.real
u_grid.set_active_scalars("cm")
u_plotter.add_mesh(u_grid, show_edges=True)

u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
```

```{code-cell} ipython3
ct_plotter = pyvista.Plotter()
ct_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
ct_grid.point_data["ct"] = ct_post.x.array.real
ct_grid.set_active_scalars("ct")
ct_plotter.add_mesh(ct_grid, show_edges=True)


ct_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    ct_plotter.show()
```
