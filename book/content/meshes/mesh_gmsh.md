---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---

# Mesh with GMSH

+++

[GMSH](https://gmsh.info/) is a powerful mesh generation tool that can be used to create complex geometries for FESTIM simulations. It supports a wide range of shapes, physical labels, and CAD import/export, making it ideal for defining detailed 2D or 3D domains.

In this tutorial, we will cover:

- Using GMSH directly from a Python script
- Converting a GMSH model into a `dolfinx` mesh that can be used with FESTIM
- Generating a mesh from a CAD geometry (e.g. STEP file)

```{admonition} Tip
:class: tip
GMSH can be installed with `conda install -c conda-forge python-gmsh`
```

+++

## DFG 3D example

This GMSH example is taken directly from [Jørgen Dokken’s GMSH tutorial](https://jsdokken.com/src/tutorial_gmsh.html).

The geometry corresponds to the domain used in the [DFG 3D CFD benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d.html) case. While we do not explore the physics of the benchmark here, the example serves as a practical demonstration of how to script mesh generation in GMSH, convert the resulting geometry into a DOLFINx mesh, and use it in a FESTIM simulation.

```{code-cell} ipython3
:tags: [hide-output]

import gmsh
import numpy as np
import os

# Initialize the GMSH API
gmsh.initialize()
gmsh.model.add("DFG 3D")

# Define geometry parameters (length L, breadth B, height H, cylinder radius r)
L, B, H, r = 2.5, 0.41, 0.41, 0.05

# Create the main channel as a rectangular box
channel = gmsh.model.occ.addBox(0, 0, 0, L, B, H)

# Create the obstacle cylinder inside the channel
cylinder = gmsh.model.occ.addCylinder(0.5, 0, 0.2, 0, B, 0, r)

# Subtract cylinder from channel to get the fluid region
fluid = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)])
gmsh.model.occ.synchronize()

# Mark the fluid volume for later identification
volumes = gmsh.model.getEntities(dim=3)
fluid_marker = 11
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid volume")

# Identify and tag boundary surfaces based on their center of mass
surfaces = gmsh.model.occ.getEntities(dim=2)
inlet, outlet = None, None
walls, obstacles = [], []

inlet_marker, outlet_marker = 1, 3
wall_marker, obstacle_marker = 5, 7

for dim, tag in surfaces:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    if np.allclose(com, [0, B / 2, H / 2]):
        gmsh.model.addPhysicalGroup(dim, [tag], inlet_marker)
        gmsh.model.setPhysicalName(dim, inlet_marker, "Fluid inlet")
        inlet = tag
    elif np.allclose(com, [L, B / 2, H / 2]):
        gmsh.model.addPhysicalGroup(dim, [tag], outlet_marker)
        gmsh.model.setPhysicalName(dim, outlet_marker, "Fluid outlet")
    elif np.isclose(com[2], 0) or np.isclose(com[1], B) or \
         np.isclose(com[2], H) or np.isclose(com[1], 0):
        walls.append(tag)
    else:
        obstacles.append(tag)

# Tag wall and obstacle surfaces
gmsh.model.addPhysicalGroup(2, walls, wall_marker)
gmsh.model.setPhysicalName(2, wall_marker, "Walls")
gmsh.model.addPhysicalGroup(2, obstacles, obstacle_marker)
gmsh.model.setPhysicalName(2, obstacle_marker, "Obstacle")

# Define mesh size field to refine near the obstacle
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)
resolution = r / 10
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

# Optionally refine mesh near inlet
inlet_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])
inlet_thre = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", 5 * resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", 10 * resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

# Apply the minimal field combining both refinement regions
minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

# Synchronize and generate 3D mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)


# Ensure the output folder exists
os.makedirs("gmsh", exist_ok=True)

# Save the mesh in GMSH format for downstream use
gmsh.write("gmsh/mesh3D.msh")
```

### Reading GMSH models

DOLFINx provides convenient tools to convert GMSH models directly into DOLFINx meshes and associated mesh tags, which can then be used within FESTIM.

The function `gmshio.model_to_mesh()` takes a GMSH model object and converts it into a DOLFINx mesh along with cell and facet markers. This is useful when working directly with GMSH from a Python script, without writing intermediate files.

```{code-cell} ipython3
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

model_rank = 0
mesh_data = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank
)
```

Alternatively, if you have saved your mesh to a `.msh` file, you can load it later using `gmshio.read_from_msh()`, specifying the mesh dimension (`gdim`).

```{code-cell} ipython3
:tags: [hide-output]

mesh_data = gmshio.read_from_msh(
    "gmsh/mesh3D.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
facet_tags = mesh_data.facet_tags
facet_tags.name = "Facet markers"

assert mesh_data.cell_tags is not None
cell_tags = mesh_data.cell_tags
cell_tags.name = "Cell markers"
```

After loading the mesh and mesh tags, you can inspect the unique identifiers assigned to cells and facets by printing their values. This helps verify that physical groups have been correctly imported.

```{code-cell} ipython3
print(f"Cell tags: {np.unique(cell_tags.values)}")
print(f"Facet tags: {np.unique(facet_tags.values)}")
```

You can visualise the mesh along with the cell and facet tags using PyVista. This provides an intuitive way to inspect the mesh structure and verify that subdomains and boundaries are correctly marked.

```{code-cell} ipython3
:tags: [hide-input]

from dolfinx import plot
import pyvista

pyvista.set_jupyter_backend("html")


tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```

```{code-cell} ipython3
:tags: [hide-input]

fdim = mesh.topology.dim - 1
tdim = mesh.topology.dim
mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(mesh, fdim, facet_tags.indices)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = facet_tags.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("facet_marker.png")
p.show()
```

```{code-cell} ipython3
:tags: [hide-input]

topology, cell_types, x = plot.vtk_mesh(mesh, tdim, cell_tags.indices)
p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Cell Marker"] = cell_tags.values
grid.set_active_scalars("Cell Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("cell_marker.png")
p.show()
```

### FESTIM Model Setup

We now present a complete FESTIM simulation using the mesh generated from GMSH.

The steady-state diffusion equation to solve is:

$$
    \nabla \cdot (D \nabla c) = 0
$$

where the diffusion coefficient, $D=1$.

The Dirichlet boundary conditions are applied as follows:

$$
    c = 1 \quad \text{on} \ \Gamma_{\mathrm{top}}
$$

$$
    c = 2 \quad \text{on} \ \Gamma_{\mathrm{bottom}}
$$

$$
    c = 0 \quad \text{on} \ \Gamma_{\mathrm{obstacle}}
$$

Here, $\Gamma_{\mathrm{top}}$, $\Gamma_{\mathrm{bottom}}$, and $\Gamma_{\mathrm{obstacle}}$ correspond to the physical boundaries marked in the mesh.

```{code-cell} ipython3
import festim as F

material = F.Material(D_0=1, E_D=0)

top_volume = F.VolumeSubdomain(id=11, material=material)

tube_surf = F.SurfaceSubdomain(id=7)
walls = F.SurfaceSubdomain(id=5)
top_surface = F.SurfaceSubdomain(id=1)
bottom_surface = F.SurfaceSubdomain(id=3)

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(mesh)

# we need to pass the meshtags to the model directly
my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

my_model.subdomains = [top_surface, bottom_surface, tube_surf, walls, top_volume]

H = F.Species("H")
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=tube_surf, value=0, species=H),
    F.FixedConcentrationBC(subdomain=top_surface, value=1, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=2, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

### Visualisation

```{code-cell} ipython3
:tags: [hide-input]

hydrogen_concentration = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(hydrogen_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = hydrogen_concentration.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

## Import CAD in GMSH

For complex geometries, GMSH allows importing CAD files such as STEP or IGES formats. 

In this example, we use a CAD model from the GMSH tutorial, generate a mesh from it, and then import the mesh into a FESTIM simulation.

```{code-cell} ipython3
:tags: [hide-output]

import gmsh
import os

gmsh.initialize()

# download cad from https://gitlab.onelab.info/gmsh/gmsh/-/raw/gmsh_4_8_4/tutorial/t20_data.step?inline=false
import requests

if not os.path.exists(os.path.join(os.pardir, "gmsh/t20_data.step")):
    url = "https://gitlab.onelab.info/gmsh/gmsh/-/raw/gmsh_4_8_4/tutorial/t20_data.step?inline=false"
    response = requests.get(url)
    with open("gmsh/t20_data.step", "wb") as f:
        f.write(response.content)

gmsh.model.add("t20")
v = gmsh.model.occ.importShapes("gmsh/t20_data.step")

gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)
vol_marker = 1
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], vol_marker)
gmsh.model.setPhysicalName(volumes[0][0], vol_marker, "Volume")

surfaces = gmsh.model.occ.getEntities(dim=2)
gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)
gmsh.model.setPhysicalName(2, 1, "Surf1")

gmsh.model.addPhysicalGroup(2, [surfaces[3][1]], 2)
gmsh.model.setPhysicalName(2, 2, "Surf2")

# Finally, let's specify a global mesh size and mesh the partitioned model:
gmsh.option.setNumber("Mesh.MeshSizeMin", 3)
gmsh.option.setNumber("Mesh.MeshSizeMax", 3)
gmsh.model.mesh.generate(3)
gmsh.write("gmsh/t20.msh")
gmsh.finalize()
```

### FESTIM model

```{code-cell} ipython3
model_rank = 0
mesh_data = gmshio.read_from_msh(
    "gmsh/t20.msh", MPI.COMM_WORLD, model_rank, gdim=3
)

mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
facet_tags = mesh_data.facet_tags
facet_tags.name = "Facet markers"
assert mesh_data.cell_tags is not None
cell_tags = mesh_data.cell_tags
cell_tags.name = "Cell markers"

print(f"Cell tags: {np.unique(cell_tags.values)}")
print(f"Facet tags: {np.unique(facet_tags.values)}")

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(mesh)

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain(id=1, material=material)

surf1 = F.SurfaceSubdomain(id=1)
surf2 = F.SurfaceSubdomain(id=2)

# we need to pass the meshtags to the model directly
my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

my_model.subdomains = [surf1, surf2, vol]

H = F.Species("H")
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=surf1, value=1, species=H),
    F.FixedConcentrationBC(subdomain=surf2, value=0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

### Visualisation

```{code-cell} ipython3
:tags: [hide-input]

hydrogen_concentration = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(hydrogen_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = hydrogen_concentration.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

## Meshing a multi-volume geometry

This section discusses how to mesh multiple volumes in GMSH, which users may need to do for a multi-material hydrogen transport problem in FESTIM. We use CadQuery, a powerful Python-based library that can build 3D parametric models. See its [documentation](https://cadquery.readthedocs.io/en/latest/intro.html) to learn more.

First, we use CadQuery to create a U-shaped pipe and then export to `.brep`:

+++

```
import cadquery as cq

tube_r = 5      # inner radius
wall = 2         # wall thickness
leg_length = 40  # vertical leg length
bend_r = 15      # centerline bend radius

path = (
    cq.Workplane("XZ")
    .vLine(leg_length)
    .radiusArc((2*bend_r, leg_length), bend_r)  # 180° bend
    .vLine(-leg_length)
    .consolidateWires()
)

# Inner (fluid) volume
inner = (
    cq.Workplane("XY")
    .circle(tube_r)
    .sweep(path)
)

# Outer solid
outer = (
    cq.Workplane("XY")
    .circle(tube_r + wall)
    .sweep(path)
)

# Tube wall
shell = outer.cut(inner)

assembly = cq.Assembly()
assembly.add(shell, name="wall", color=cq.Color("gray"))
assembly.add(inner, name="fluid", color=cq.Color("blue"))

assembly.toCompound().exportBrep("tube.brep")
```

+++

This should create an output file named `tube.brep`.

```{note}
In this tutorial we do not run the code section above, since `cadquery` and `dolfinx.0.10` has a dependency conflict. We recommend having a separate environment to run CadQuery, as shown in [this repo](https://github.com/kaelyndunnell/Fusion-TES-Modeling).
```

Now, we mesh the CAD in GMSH. First, we add the CAD and get the volumes/surfaces using `getEntities` (3 for volumes, 2 for surfaces):

```{code-cell} ipython3
import gmsh

gmsh.initialize()
gmsh.option.setString("Geometry.OCCTargetUnit", "M")
gmsh.model.add("tube")

gmsh.merge("tube.brep")
gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)

print(f"Found {len(volumes)} volumes")
print(f"Found {len(surfaces)} surfaces")
```

```{tip}
For complex geometries, it may be helpful to run `gmsh.model.occ.fragment` to break up volumes and splitting overlapping entities. We do not use it here since our geometry is fairly simple.
```

+++

Here we separate individual volumes and surfaces for physical grouping:

```{code-cell} ipython3
fluid_vol = volumes[1][1]
wall_vol  = volumes[0][1]

outer_surfaces = [1, 2, 3, 5, 7]
interfaces     = [4, 6, 8]
inlet          = [9]
outlet         = [10]

# Setting markers

fluid_vol_marker = 2
wall_vol_marker = 1
outer_surfaces_marker = 10
interfaces_marker = 11
inlet_marker = 12
outlet_marker = 13
```

```{tip}
You can find these surface IDs by running the GMSH script without adding physical groups. Once you create your mesh, open it in GMSH and go to **Tools->Visibility** to get the correct IDs for the corresponding surfaces and volumes. Then, use those tags in your GMSH script to make physical groups.
```

+++

Let's assign physical groups and markers:

```{code-cell} ipython3
# Volumes
gmsh.model.addPhysicalGroup(3, [fluid_vol], tag=fluid_vol_marker, name="fluid")
gmsh.model.addPhysicalGroup(3, [wall_vol], tag=wall_vol_marker, name="wall")

# Surfaces
gmsh.model.addPhysicalGroup(2, outer_surfaces, tag=outer_surfaces_marker, name="outer_surfaces")
gmsh.model.addPhysicalGroup(2, interfaces, tag=interfaces_marker, name="fluid_wall_interface")
gmsh.model.addPhysicalGroup(2, inlet, tag=inlet_marker, name="inlet")
gmsh.model.addPhysicalGroup(2, outlet, tag=outlet_marker, name="outlet")
```

For curved meshes, users can use `MeshSizeFromCurvature` to specify element sizes (higher values lead to more refinement):

```{code-cell} ipython3
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
```

Now we generate the mesh:

```{code-cell} ipython3
:tags: [hide-output]

gmsh.model.mesh.generate(3)
gmsh.write("tube.msh")
gmsh.finalize()
```

We should have a new file named `tube.msh`. Similar to the example above, we can read the mesh information for DOLFINx:

```{code-cell} ipython3
from mpi4py import MPI
import festim as F

mesh_data = gmshio.read_from_msh(
    "tube.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh = mesh_data.mesh
facet_tags = mesh_data.facet_tags
facet_tags.name = "Facet markers"

cell_tags = mesh_data.cell_tags
cell_tags.name = "Cell markers"
```

```{code-cell} ipython3
:tags: [hide-input]

from dolfinx import plot
import pyvista

fdim = mesh.topology.dim - 1
tdim = mesh.topology.dim
mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(mesh, fdim, facet_tags.indices)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = facet_tags.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("facet_marker.png")
p.show()
```

We can now use this mesh in FESTIM for a discontinuous problem:

```{code-cell} ipython3
my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh(mesh)

fluid = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
wall = F.Material(D_0=0.01, E_D=0, K_S_0=2, E_K_S=0)

fluid_vol = F.VolumeSubdomain(id=fluid_vol_marker, material=fluid)
wall_vol = F.VolumeSubdomain(id=wall_vol_marker, material=wall)

inlet = F.SurfaceSubdomain(id=inlet_marker)
outlet = F.SurfaceSubdomain(id=outlet_marker)
outer_surfaces = F.SurfaceSubdomain(id=outer_surfaces_marker)

my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

my_model.subdomains = [inlet, outlet, outer_surfaces, fluid_vol, wall_vol]
my_model.surface_to_volume = {
    outer_surfaces: wall_vol,
    inlet: fluid_vol,
    outlet: fluid_vol
}
my_model.interfaces =[
    F.Interface(id=interfaces_marker, subdomains=[wall_vol, fluid_vol], penalty_term=1e5)
]
H = F.Species("H", subdomains=[fluid_vol, wall_vol])
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=inlet, value=1, species=H),
    F.FixedConcentrationBC(subdomain=outlet, value=0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
my_model.initialise()
my_model.run()
```

```{code-cell} ipython3
:tags: [hide-input]

import pyvista 
from dolfinx import plot

def make_ugrid(solution):
    topology, cell_types, geometry = plot.vtk_mesh(solution.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = solution.x.array.real
    u_grid.set_active_scalars("c")
    return u_grid

pyvista.set_jupyter_backend("html")

u_plotter = pyvista.Plotter()
u_grid_fluid = make_ugrid(H.subdomain_to_post_processing_solution[fluid_vol])
u_grid_wall = make_ugrid(H.subdomain_to_post_processing_solution[wall_vol])
u_plotter.add_mesh(u_grid_fluid, cmap="viridis", show_edges=False)
u_plotter.add_mesh(u_grid_wall, cmap="viridis", show_edges=False)
u_plotter.view_yx()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```

```{code-cell} ipython3
:tags: [hide-input]

pyvista.set_jupyter_backend("html")

u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid_fluid, cmap="viridis", show_edges=False)
u_plotter.view_yx()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```
