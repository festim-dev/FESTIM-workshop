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

# Training a FESTIM surrogate model

```{code-cell} ipython3
import festim as F

from dolfinx.mesh import create_unit_square
from mpi4py import MPI


def make_model(c_boundary: float, source: float) -> F.HydrogenTransportProblem:
    fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

    festim_mesh = F.Mesh(fenics_mesh)

    material_top = F.Material(D_0=2, E_D=0)
    material_bot = F.Material(D_0=1, E_D=0)

    top_volume = F.VolumeSubdomain(
        id=1, material=material_top, locator=lambda x: x[1] >= 0.5
    )
    bottom_volume = F.VolumeSubdomain(
        id=2, material=material_bot, locator=lambda x: x[1] <= 0.5
    )

    boundary = F.SurfaceSubdomain(id=1)

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = festim_mesh
    my_model.subdomains = [boundary, top_volume, bottom_volume]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = 400

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=boundary, value=c_boundary, species=H),
    ]

    my_model.sources = [F.ParticleSource(species=H, volume=bottom_volume, value=source)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [
        F.TotalVolume(field=H, volume=top_volume),
        F.TotalVolume(field=H, volume=bottom_volume),
    ]

    return my_model
```

```{code-cell} ipython3
import festim as F
from autoemulate.simulations.base import Simulator
import torch


class FestimProblem(Simulator):
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        c_boundary = x[:, 0]
        source = x[:, 1]

        # convert to float
        c_boundary = c_boundary.item()
        source = source.item()
        model = make_model(c_boundary, source)

        # Solve the model
        model.initialise()
        model.run()

        # Extract the total amount of H in the top and bottom volumes
        total_top = model.exports[0].data
        total_bot = model.exports[1].data


        y = torch.tensor([total_top, total_bot]).T
        # Ensure the output is a 2D tensor
        if y.ndim == 1:
            y = y.unsqueeze(1)
        
        return y
```

```{code-cell} ipython3
model = make_model(1.0, 1.0)
model.initialise()
model.run()
```

```{code-cell} ipython3
from dolfinx import plot
import pyvista

def make_ugrid(solution):
    topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = solution.x.array.real
    u_grid.set_active_scalars("c")
    return u_grid

pyvista.set_jupyter_backend("html")

u_plotter = pyvista.Plotter()

H = model.species[0]

u_grid = make_ugrid(H.post_processing_solution)
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.view_xy()
u_plotter.add_text("Hydrogen concentration in multi-material problem", font_size=12)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

```{code-cell} ipython3
simulator = FestimProblem(parameters_range={'c_boundary': (0.0, 1.0), 'source': (0.0, 100.0)}, output_names=['total_top', 'total_bot'])
```

```{code-cell} ipython3
simulator.forward(torch.tensor([[0.0, 3.0]]))
```

```{code-cell} ipython3
n_samples = 80

X = simulator.sample_inputs(n_samples)

X.shape
```

```{code-cell} ipython3
Y, _ = simulator.forward_batch(X, allow_failures=False)
Y.shape
```

```{code-cell} ipython3
Y
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for i in range(2):
    plt.sca(axs[i])
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, i], cmap='viridis')

    plt.colorbar()
    plt.title(f'{simulator.output_names[i]}')

    plt.xlabel(f"{list(simulator.parameters_range.keys())[0]}")
    plt.ylabel(f"{list(simulator.parameters_range.keys())[1]}")


plt.show()
```

```{code-cell} ipython3
from autoemulate import AutoEmulate
# Run AutoEmulate with default settings
ae = AutoEmulate(X, Y, log_level="info")
```

```{code-cell} ipython3
ae.summarise()
```

```{code-cell} ipython3
# best = ae.results[0]
best = ae.best_result()
print("Model with id: ", best.id, " performed best: ", best.model_name)
```

```{code-cell} ipython3
ae.plot_preds(best, output_names=simulator.output_names)
```

```{code-cell} ipython3
from autoemulate.core.plotting import create_and_plot_slice

for i in range(2):

    fig, axs = create_and_plot_slice(
        best.model,
        output_idx=i,
        parameters_range=simulator.parameters_range,
        quantile=0.5,
        param_pair=(0, 1),
    )
    plt.scatter(X[:, 0], X[:, 1])
    plt.suptitle(f'{simulator.output_names[i]} - slice plot')
    plt.show()
```
