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

# DOLFINx meshes

DOLFINx provides a range of simple built-in meshes that are useful for quick testing and solving basic problems. You can find more information about the available mesh types, as well as tools for creating, refining, and marking meshes, in the [DOLFINx mesh documentation](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/generated/dolfinx.mesh.html).

In this tutorial, we will cover the following:

- How to create basic meshes using DOLFINx  
- How to apply mesh transformations  
- How to use these meshes in a FESTIM simulation  
- How to define and label subdomains on DOLFINx meshes  

+++

## Built-in DOLFINx meshes

The `create_unit_square` function in DOLFINx generates a structured, 2D mesh of the unit square, divided into a specified number of cells in the x and y directions. This is useful for simple test cases and provides a convenient starting point for setting up simulations.

```{code-cell} ipython3
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

nx, ny = 10, 10  # Number of divisions in x and y directions  
mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)
```

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
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```

Similarly, the `create_unit_cube` function creates a 3D mesh of the unit cube, subdivided into a specified number of cells along the x, y, and z directions. It is useful for testing and developing 3D simulations.

```{code-cell} ipython3
from dolfinx.mesh import create_unit_cube
from mpi4py import MPI

nx, ny, nz = 10, 10, 10  # Number of divisions in x, y, and z directions
mesh = create_unit_cube(MPI.COMM_WORLD, nx, ny, nz)
```

```{code-cell} ipython3
:tags: [hide-input]

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")    
```

The `create_rectangle` function enables the definition of a 2D rectangular mesh over any axis-aligned domain by specifying custom corner coordinates, unlike `create_unit_square`, which has a fixed domain. This provides more flexibility for modelling geometries of arbitrary size.

```{code-cell} ipython3
from dolfinx.mesh import create_rectangle
from mpi4py import MPI

mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [2, 1]], [20, 10])
```

```{code-cell} ipython3
:tags: [hide-input]

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```

All of the built-in mesh creation functions in DOLFINx, such as `create_unit_square`, `create_unit_cube`, and `create_rectangle`, accept a `cell_type` argument. This specifies the type of elements used to construct the mesh.

Common options include:

- `"triangle"` for 2D triangular elements  
- `"quadrilateral"` for 2D quadrilateral elements  
- `"tetrahedron"` for 3D tetrahedral elements  
- `"hexahedron"` for 3D hexahedral (brick-like) elements

Choosing the appropriate `cell_type` is important, as it determines the structure of the mesh and which finite elements can be used later in the simulation.

```{code-cell} ipython3
from dolfinx.mesh import CellType

mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=CellType.quadrilateral)
```

## Mesh transformations

In some cases, it is useful to apply transformations to a mesh's coordinates, such as scaling, translating, or rotating the domain. This can help with unit conversion, adjusting the position of the mesh, or aligning it with physical geometry.

These operations can be performed directly by modifying the `.geometry.x` attribute of a DOLFINx mesh, which stores the coordinate array of all mesh vertices.

### Scaling

To scale a mesh uniformly or anisotropically, you can multiply the coordinate array by a scalar or a vector. This changes the physical size of the domain without altering its topological structure.

```{code-cell} ipython3
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

mesh.geometry.x[:] *= 1e-3
```

The following example shows how to scale only the x-dimension of a mesh, effectively compressing the domain in one direction:

```{code-cell} ipython3
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

mesh.geometry.x[:, 0] *= 0.5
```

This creates a 2D unit square mesh with 10 divisions in each direction. The second line accesses the mesh’s geometry array and modifies only the first column, which corresponds to the x-coordinates of all vertices. Multiplying these values by 0.5 scales the mesh in the x-direction, transforming the original square into a rectangle. This approach is useful when you need to stretch or compress a mesh in a single direction, for example to simulate an anisotropic domain or convert units along one axis.

```{code-cell} ipython3
:tags: [hide-input]

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```

### Translating

To move (translate) a mesh to a different position, simply add a constant offset to all coordinates:

```{code-cell} ipython3
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

# Move mesh 10 units in x-direction
mesh.geometry.x[:, 0] += 10
```

### Rotating

#### Example 1: Rotating a mesh by 90 Degrees by swapping coordinates

This method rotates a 2D mesh 90 degrees counter-clockwise by swapping its x- and y-coordinates.

The mesh is originally defined over the domain $([0, 2] \times [0, 1])$. After swapping coordinates, the domain becomes $([0, 1] \times [0, 2])$, effectively rotating the mesh about the origin.

This approach is simple and efficient but limited to 90° rotations and rotation about the origin.

```{note} Note
The rotation is performed about the origin $(0, 0)$. If your mesh is defined elsewhere, you may need to shift it to the origin before rotation and shift it back afterward.
```

```{code-cell} ipython3
# Create a rectangular mesh over the domain [0, 2] x [0, 1]
mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [2, 1]], [20, 10])

# Store a copy of the x-coordinates
x_vals = mesh.geometry.x[:, 0].copy()

# Swap x and y to perform a 90-degree rotation
mesh.geometry.x[:, 0] = mesh.geometry.x[:, 1].copy()
mesh.geometry.x[:, 1] = x_vals
```

```{code-cell} ipython3
:tags: [hide-input]

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()

# show orientation
plotter.show_axes()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```

#### Example 2: Rotating a mesh by an arbitrary angle

To rotate a mesh by any angle, you can apply a rotation matrix using the `Rotation` class from `scipy.spatial.transform`.

In this example, the mesh is rotated by 30 degrees counter-clockwise about the origin. The rotation is performed by applying the rotation matrix to all mesh coordinates.

If your mesh is not centred at the origin, you should translate it so the rotation point aligns with the origin before rotating, and translate it back afterward.

```{code-cell} ipython3
from scipy.spatial.transform import Rotation

mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

# Define rotation angle in degrees
degrees = 30

# Create a rotation object for rotation about the z-axis
rotation = Rotation.from_euler("z", degrees, degrees=True)

# Apply rotation to all mesh coordinates
mesh.geometry.x[:, :] = rotation.apply(mesh.geometry.x)
```

```{code-cell} ipython3
:tags: [hide-input]

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()

# show orientation
plotter.show_axes()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
```

## Passing a DOLFINx mesh to FESTIM

To use a DOLFINx mesh in a FESTIM simulation, simply pass it to a `festim.Mesh` object. This allows FESTIM to internally work with the same mesh data structure used by DOLFINx.

You can pass any 2D or 3D DOLFINx mesh, created manually or through `create_unit_square`, `create_rectangle`, `create_unit_cube`, etc., directly into FESTIM.

```{code-cell} ipython3
import festim as F

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

festim_mesh = F.Mesh(fenics_mesh)
```

## Defining subdomains

Now that we know how to pass a DOLFINx mesh to FESTIM, the next step is to define subdomains. These are required to assign different materials, boundary conditions, or sources to specific regions of the mesh.

In FESTIM, subdomains can be defined on:

- **Volumes** (used for material properties, sources, etc.)
- **Surfaces** (used for boundary conditions like Dirichlet, Neumann, or Robin)

### Surface subdomains

To define surface (boundary) subdomains in FESTIM, we make use of the `festim.SurfaceSubdomain` class.

Here's an example for the top and bottom boundaries of a 2D mesh:

```{code-cell} ipython3
import numpy as np

# Instantiate the surface subdomains with unique IDs
top_surface = F.SurfaceSubdomain(id=1, locator = lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator = lambda x: np.isclose(x[1], 0.0))
```

### Volume subdomains

Similarly to surface subdomains, FESTIM allows you to define volume subdomains by subclassing `festim.VolumeSubdomain`.

Volume subdomains are used to assign different materials or define regions with specific physical properties. Each volume subdomain must be associated with a `festim.Material` object.

In the example below, we divide a mesh into two regions along the y-axis: the **top half** $(y \geq 0.5)$ and the **bottom half** $(y \leq 0.5)$. Both subdomains are assigned the same test material.

```{code-cell} ipython3
# volume subdomains need a material
material = F.Material(name="test_material", D_0=1, E_D=0)

# Instantiate the volume subdomains, both using the same material
top_volume = F.VolumeSubdomain(id=1, material=material, locator=lambda x: x[1] >= 0.5)
bottom_volume = F.VolumeSubdomain(id=2, material=material, locator=lambda x: x[1] <= 0.5)
```

### Visualising subdomains

Once surface and volume subdomains are defined, they can be visualised by passing them to a FESTIM model and calling the `define_meshtags_and_measures()` method.

This method internally generates `dolfinx.mesh.MeshTags` objects for both volume and facet entities. These tags associate mesh entities with their corresponding subdomain IDs, making them suitable for visualisation or for use in boundary conditions and material assignment.

You can access the generated mesh tags using:

- `my_model.facet_meshtags` for surface subdomains
- `my_model.volume_meshtags` for volume subdomains

```{code-cell} ipython3
my_model = F.HydrogenTransportProblem()
my_model.mesh = festim_mesh

# Assign subdomains (surface and volume)
my_model.subdomains = [top_surface, bottom_surface, top_volume, bottom_volume]

# Generate dolfinx.MeshTags for visualisation and boundary/material definition
my_model.define_meshtags_and_measures()

print(my_model.facet_meshtags)
print(my_model.volume_meshtags)
```

After calling `define_meshtags_and_measures()` on your FESTIM model, you can use `pyvista` to visualise both **volume subdomains** and **surface subdomains**. 

This is especially helpful for verifying that subdomains are correctly defined before running a simulation.

The examples below show how to extract the mesh data and associated subdomain IDs and render them using PyVista. The resulting mesh will highlight different subdomains using distinct colours.

```{code-cell} ipython3
fdim = my_model.mesh.mesh.topology.dim - 1
tdim = my_model.mesh.mesh.topology.dim
my_model.mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    my_model.mesh.mesh, tdim, my_model.volume_meshtags.indices
)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Cell Marker"] = my_model.volume_meshtags.values
grid.set_active_scalars("Cell Marker")
p.add_mesh(grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("volume_markers.png")
```

```{code-cell} ipython3
my_model.mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    my_model.mesh.mesh, fdim, my_model.facet_meshtags.indices
)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = my_model.facet_meshtags.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_markers.png")
```

To reuse subdomain information across different scripts or simulation runs, you can save your `MeshTags` to a file using the `io4dolfinx` library. This is particularly useful for workflows where subdomain definition is done separately from the main simulation.

The example below shows how to store surface (facet) subdomains to a `.bp` file and read them back later. You can follow the same procedure for volume subdomains.

Make sure to assign a unique name to the `MeshTags` before writing, as this is required for identifying them when reading.

```{code-cell} ipython3
import io4dolfinx

my_model.facet_meshtags.name = "facet_tags"

# write
io4dolfinx.write_meshtags("facet_tags.bp", my_model.mesh.mesh, my_model.facet_meshtags)

# read
ft = io4dolfinx.read_meshtags("facet_tags.bp", my_model.mesh.mesh, meshtag_name="facet_tags")
```

## Complete example

### Implementation

```{code-cell} ipython3
import festim as F
import numpy as np

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

festim_mesh = F.Mesh(fenics_mesh)

material_top = F.Material(D_0=1, E_D=0)
material_bot = F.Material(D_0=2, E_D=0)


top_volume = F.VolumeSubdomain(id=1, material=material_top, locator=lambda x: x[1] >= 0.5)
bottom_volume = F.VolumeSubdomain(id=2, material=material_bot, locator=lambda x: x[1] <= 0.5)

top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))

my_model = F.HydrogenTransportProblem()
my_model.mesh = festim_mesh
my_model.subdomains = [top_surface, bottom_surface, top_volume, bottom_volume]

H = F.Species("H")
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()
```

### Visualisation

```{code-cell} ipython3
hydrogen_concentration = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(hydrogen_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = hydrogen_concentration.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")
```
