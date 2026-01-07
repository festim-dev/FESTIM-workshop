---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

# Monoblock #

In this task, we will learn how to run CAD-based simulations.

Our example case will be a 3D ITER-like monoblock made of three different materials (tungsten, cucrzr, and copper).

```{code-cell} ipython3
import meshio


def convert_med_to_xdmf(
    med_file,
    cell_file="mesh_domains.xdmf",
    facet_file="mesh_boundaries.xdmf",
    cell_type="tetra",
    facet_type="triangle",
):
    """Converts a MED mesh to XDMF
    Args:
        med_file (str): the name of the MED file
        cell_file (str, optional): the name of the file containing the
            volume markers. Defaults to "mesh_domains.xdmf".
        facet_file (str, optional): the name of the file containing the
            surface markers.. Defaults to "mesh_boundaries.xdmf".
        cell_type (str, optional): The topology of the cells. Defaults to "tetra".
        facet_type (str, optional): The topology of the facets. Defaults to "triangle".
    Returns:
        dict, dict: the correspondance dict, the cell types
    """
    msh = meshio.read(med_file)

    correspondance_dict = {-k: v for k, v in msh.cell_tags.items()}
    
    cell_data_types = msh.cell_data_dict["cell_tags"].keys()

    for mesh_block in msh.cells:
        if mesh_block.type == cell_type:
            meshio.write_points_cells(
                cell_file,
                msh.points,
                [mesh_block],
                cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][cell_type]]},
            )
        elif mesh_block.type == facet_type:
            meshio.write_points_cells(
                facet_file,
                msh.points,
                [mesh_block],
                cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][facet_type]]},
            )

    return correspondance_dict, cell_data_types
```

```{code-cell} ipython3
correspondance_dict, cell_data_types = convert_med_to_xdmf(
    "monoblock_mesh/mesh.med",
    cell_file="monoblock_mesh/mesh_domains.xdmf",
    facet_file="monoblock_mesh/mesh_boundaries.xdmf",
)

print(correspondance_dict)
```

```{code-cell} ipython3
import festim as F

tungsten = F.Material(
    D_0=4.1e-7,
    E_D=0.39,
    K_S_0=1.87e24,
    E_K_S=1.04,
    thermal_conductivity=100,
)

copper = F.Material(
    D_0=6.6e-7,
    E_D=0.387,
    K_S_0=3.14e24,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0=3.92e-7, E_D=0.418, K_S_0=4.28e23, E_K_S=0.387, thermal_conductivity=350
)
```

The converted .xdmf files can then be imported in FESTIM using the `MeshFromXDMF` class:

```{code-cell} ipython3
mesh = F.MeshFromXDMF(
    volume_file="monoblock_mesh/mesh_domains.xdmf", facet_file="monoblock_mesh/mesh_boundaries.xdmf"
)

mesh.mesh.geometry.x[:] *= 1e-3  # mm to m
```

```{code-cell} ipython3
tungsten_volume = F.VolumeSubdomain(id=6, material=tungsten)
copper_volume = F.VolumeSubdomain(id=7, material=copper)
cucrzr_volume = F.VolumeSubdomain(id=8, material=cucrzr)
top_surface = F.SurfaceSubdomain(id=9)
cooling_surface = F.SurfaceSubdomain(id=10)
poloidal_gap_w = F.SurfaceSubdomain(id=11)
poloidal_gap_cu = F.SurfaceSubdomain(id=12)
poloidal_gap_cucrzr = F.SurfaceSubdomain(id=13)
toroidal_gap = F.SurfaceSubdomain(id=14)
bottom = F.SurfaceSubdomain(id=15)

all_subdomains = [
    tungsten_volume,
    copper_volume,
    cucrzr_volume,
    top_surface,
    cooling_surface,
    poloidal_gap_cu,
    poloidal_gap_w,
    poloidal_gap_cucrzr,
    toroidal_gap,
    bottom,
]
```

```{code-cell} ipython3
heat_transfer_problem = F.HeatTransferProblem()
heat_transfer_problem.subdomains = all_subdomains
heat_transfer_problem.mesh = mesh

heat_flux_top = F.HeatFluxBC(subdomain=top_surface, value=10e6)
convective_flux_coolant = F.HeatFluxBC(
    subdomain=cooling_surface, value=lambda T: 7e04 * (323 - T)
)


heat_transfer_problem.boundary_conditions = [heat_flux_top, convective_flux_coolant]

heat_transfer_problem.exports = [F.VTXTemperatureExport("out.bp")]

heat_transfer_problem.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)
heat_transfer_problem.initialise()
heat_transfer_problem.run()
```

```{code-cell} ipython3
my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.method_interface = "penalty"

my_model.subdomains = all_subdomains

H = F.Species("H", subdomains=my_model.volume_subdomains)
my_model.species = [H]

my_model.mesh = mesh

my_model.surface_to_volume = {
    top_surface: tungsten_volume,
    cooling_surface: cucrzr_volume,
    poloidal_gap_w: tungsten_volume,
    poloidal_gap_cu: copper_volume,
    poloidal_gap_cucrzr: cucrzr_volume,
    toroidal_gap: tungsten_volume,
    bottom: tungsten_volume,
}

penalty_term = 1e20
my_model.interfaces = [
    F.Interface(
        id=16, subdomains=(tungsten_volume, copper_volume), penalty_term=penalty_term
    ),
    F.Interface(
        id=17, subdomains=(copper_volume, cucrzr_volume), penalty_term=penalty_term
    ),
]
```

Using the tags provided by `correspondance_dict`, we can create materials and assign them to the simulation:

+++

Similarily, the surface tags are used to create boundary conditions:

```{code-cell} ipython3
import ufl

phi = 1.61e22
R_p = 9.52e-10
implantation_flux_top = F.FixedConcentrationBC(
    subdomain=top_surface,
    value=lambda T: phi * R_p / (tungsten.D_0 * ufl.exp(-tungsten.E_D / F.k_B / T)),
    species=H,
)

recombination_fluxes = [
    F.FixedConcentrationBC(subdomain=surf, value=0, species=H)
    for surf in [
        toroidal_gap,
        bottom,
        poloidal_gap_w,
        poloidal_gap_cu,
        poloidal_gap_cucrzr,
        cooling_surface,
    ]
]


my_model.boundary_conditions = [implantation_flux_top] + recombination_fluxes

my_model.temperature = 1200  # heat_transfer_problem.u  # FIXME https://fenicsproject.discourse.group/t/interpolate-expression-on-submesh-with-parent-mesh-function/17467

my_model.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    transient=False,
    max_iterations=10,
)

my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
assert my_model.interfaces[0].id in tungsten_volume.ft.values
assert my_model.interfaces[0].id in copper_volume.ft.values
assert my_model.interfaces[1].id in copper_volume.ft.values
assert my_model.interfaces[1].id in cucrzr_volume.ft.values
```

## Post processing

```{code-cell} ipython3
:tags: [hide-cell]

import pyvista

pyvista.start_xvfb()
pyvista.set_jupyter_backend("html")
```

```{code-cell} ipython3
from dolfinx import plot

T = heat_transfer_problem.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["T"] = T.x.array.real
u_grid.set_active_scalars("T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")
```

```{code-cell} ipython3
u_plotter = pyvista.Plotter()

for vol in my_model.volume_subdomains:
    sol = H.subdomain_to_post_processing_solution[vol]

    topology, cell_types, geometry = plot.vtk_mesh(sol.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = sol.x.array.real
    u_grid.set_active_scalars("c")
    u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
    u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

u_plotter.view_xy(negative=True)


if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```
