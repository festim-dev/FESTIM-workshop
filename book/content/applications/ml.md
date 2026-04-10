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


def make_model(source_bottom: float, source_top: float) -> F.HydrogenTransportProblem:
    fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

    festim_mesh = F.Mesh(fenics_mesh)

    material_top = F.Material(D_0=0.2, E_D=0)
    material_bot = F.Material(D_0=0.1, E_D=0)

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
        F.FixedConcentrationBC(subdomain=boundary, value=0.0, species=H),
    ]

    my_model.sources = [
        F.ParticleSource(species=H, volume=bottom_volume, value=source_bottom),
        F.ParticleSource(species=H, volume=top_volume, value=source_top),
    ]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.exports = [
        F.TotalVolume(field=H, volume=top_volume),
        F.TotalVolume(field=H, volume=bottom_volume),
    ]

    return my_model
```

```{code-cell} ipython3
from dolfinx import plot
import pyvista
pyvista.set_jupyter_backend("html")


def make_ugrid(solution, label="c"):
    topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data[label] = solution.x.array.real
    u_grid.set_active_scalars(label)
    return u_grid

u_plotter = pyvista.Plotter(shape=(2,2))

for i, (source_bottom, source_top) in enumerate([(0.0, 1.0), (1.0, 0.0), (4.0, 1.0), (1.0, 2.0)]):
    model = make_model(source_bottom, source_top)
    model.initialise()
    model.run()

    H = model.species[0]
    u_grid = make_ugrid(H.post_processing_solution)
    u_plotter.subplot(i // 2, i % 2)
    warped = u_grid.warp_by_scalar(factor=1)
    u_plotter.add_mesh(warped, cmap="viridis", show_edges=True)
    u_plotter.add_text(f"source_bottom={source_bottom}, source_top={source_top}", font_size=10)
    u_plotter.link_views()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

```{code-cell} ipython3
from autoemulate.simulations.base import Simulator
import torch


class FestimProblem(Simulator):
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        source_top = x[:, 0]
        source_bottom = x[:, 1]

        # convert to float
        source_top = source_top.item()
        source_bottom = source_bottom.item()
        model = make_model(source_bottom, source_top)

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
simulator = FestimProblem(parameters_range={'source_top': (0.0, 10.0), 'source_bottom': (0.0, 10.0)}, output_names=['total_top', 'total_bot'])
```

```{code-cell} ipython3
simulator.forward(torch.tensor([[0.0, 3.0]]))
```

```{code-cell} ipython3
n_samples = 20

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
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, i], cmap='viridis', vmin=Y.min(), vmax=Y.max())

    plt.title(f'{simulator.output_names[i]}')

    plt.xlabel(f"{simulator.param_names[0]}")
    plt.ylabel(f"{simulator.param_names[1]}")

plt.colorbar(cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]))

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
