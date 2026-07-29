# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FESTIM-workshop is a **Jupyter Book tutorial** (published at festim-workshop.readthedocs.io) for
[FESTIM](https://festim.readthedocs.io), a finite-element hydrogen-transport / heat-transfer code
built on FEniCSx (dolfinx). This repo is teaching material — collections of executable notebooks —
not a Python package. There is nothing to `pip install` from here and no unit test suite; "correct"
means the book builds and every notebook runs top to bottom without error or warning.

## Environment

Everything runs in the `festim-workshop` conda env (dolfinx, FEniCSx, and pip-only deps cannot be
installed with plain pip):

```bash
conda env create -f environment.yml   # first time
conda activate festim-workshop
```

Key pinned deps: `festim==2.2rc1`, `fenics-dolfinx>=0.11`, `h-transport-materials` (material
property DB), `foam2dolfinx` (OpenFOAM→dolfinx CFD coupling), `openmc2dolfinx` (neutronics),
`autoemulate`+`torch` (ML surrogates), `python-gmsh`, `pyvista`/`trame` (3D viz).

**Python must stay an exact pin (`python=3.12`), never a `<3.13`-style constraint.** repo2docker —
which builds both the Binder image and the devcontainer image — only honours the requested Python
when it is pinned exactly; with a constraint it builds its base env on its own default Python and
then rewrites the entire stack on top, which produced mixed py310/py312 builds of
`petsc4py`/`kahip`/`netcdf4` and a failed Binder build.

## Binder, live code, and Codespaces

The environment is too heavy (dolfinx + vtk + qt6 + ffmpeg + occt + torch) to resolve inside a
mybinder build, so `.github/workflows/build-image.yml` builds it once per push to `main` via
`jupyterhub/repo2docker-action` and pushes `ghcr.io/festim-dev/festim-workshop`. Consequences:

- `binder/Dockerfile` is **generated** (`FROM <image>:<sha>`) and committed by that workflow.
  Never edit it, and never add other files to `binder/` — the action aborts if it finds any.
- `apt.txt` at the repo root supplies system GL for pyvista. The blanket `*.txt` rule in
  `.gitignore` is negated by a `!apt.txt` line; keep that negation if you touch `.gitignore`.
- The workflow must stay triggered on **every** push to `main`: `binder/Dockerfile` is only a
  `FROM`, so repo2docker does not re-copy the repo and notebook edits reach Binder only by rebuild.
- The image's env lives at `/srv/conda/envs/notebook` (not `festim-workshop`) and runs as `jovyan`
  — that is what `.devcontainer/devcontainer.json` points at.
- The GHCR package must be **public** or Binder cannot pull it.

## Build the book

```bash
jupyter-book build book          # outputs to book/_build/html/index.html
```

Read the Docs reproduces the build via a `pre_build` step then Sphinx — see `.readthedocs.yml`.
Two build behaviors matter when editing content:

- **Notebooks are force-executed on every build** (`execute_notebooks: force`, `timeout: -1` in
  `book/_config.yml`). A change that makes any cell raise breaks the build. Expect long build times.
- **`fail_on_warning: true`** (in `.readthedocs.yml`): a Sphinx warning — a broken cross-reference,
  a missing image, a bad `codeautolink` — fails the RTD build even if notebooks execute fine.

Run a single notebook without a full book build to check it executes:

```bash
jupyter nbconvert --to notebook --execute book/content/applications/task02.ipynb --stdout > /dev/null
# MyST-markdown pages: convert first with jupytext, or open in Jupyter (they are paired notebooks)
jupytext --to notebook book/content/material/material_basics.md -o /tmp/out.ipynb
```

## Content structure

- `book/intro.md` — landing page; `book/_config.yml` — Jupyter Book config; `book/_toc.yml` — table
  of contents. **Any new page must be registered in `book/_toc.yml` or it won't appear.**
- `book/content/` is organized by concept, mirroring FESTIM's building blocks:
  `meshes/`, `material/`, `boundary_conditions/`, `species_reactions/`, `initial_conditions/`,
  `temperatures/`, `post_process/`, `misc/`, and `applications/` (end-to-end `taskNN` cases plus
  `multiphysics/` CFD+neutronics and `ml/` surrogate/active-learning).
- `book/content/misc/festim_from_scratch.md` + `dolfinx_*.md` show the same physics written in raw
  dolfinx rather than the FESTIM API — useful when explaining what FESTIM does under the hood.

### Page formats — two interchangeable notebook types

Every tutorial page is an **executable notebook** in one of two formats; treat both as code:

1. `.ipynb` — standard Jupyter notebooks (most `applications/taskNN`).
2. `.md` with a `jupytext` YAML header (`formats: ipynb,md:myst`) — MyST-markdown notebooks. These
   are notebooks stored as text and are executed by the build exactly like `.ipynb`. Editing the
   Python inside a fenced ```` ```{code-cell} ```` block is editing runnable code.

There are no plain prose `.md` files — every `.md` under `book/content/` is a MyST notebook.

## Conventions

- API objects in code cells auto-link to upstream docs via `sphinx-codeautolink` + `intersphinx`
  (festim, dolfinx, numpy, matplotlib…). Reference real, importable names so links resolve; a bad
  reference is a build warning.
- `.bp`, `.h5`/`.xdmf`, `.msh` files committed next to notebooks are simulation outputs / meshes that
  cells read or write — don't assume they're stale artifacts before checking the notebook that uses them.
- Note: `.github/workflows/test_notebooks.yml` still targets an old flat `tasks/` directory that no
  longer exists (content moved under `book/content/applications/`); that workflow is stale.
