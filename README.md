# FESTIM-workshop

[![Test](https://github.com/festim-dev/FESTIM-workshop/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/festim-dev/FESTIM-workshop/actions/workflows/test_notebooks.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/festim-dev/FESTIM-workshop/main)

## Tasks

### Basic

[Task 1](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task01.ipynb): Simple hydrogen transport simulation

[Task 2](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task02.ipynb): Simulation of a Thermo-Desorption experiment

[Task 3](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task03.ipynb): Simple permeation model

[Task 4](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task04.ipynb): Permeation barrier modelling

[Task 5](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task05.ipynb): Post-processing and visualisation

[Task 7](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task07.ipynb): Heat transfer simulation

[Task 8](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task08.ipynb): CAD integration

[Task 9](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task09.ipynb): Integration with the HTM library


### Advanced

[Task 6](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task06.ipynb): Advection-diffusion problem

[Task 10](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task10.ipynb): Fitting a TDS spectrum

[Task 11](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task11.ipynb): Radioactive decay ☢️

[Task 12](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task12.ipynb): Soret Effect

[Task 13](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task13.ipynb): Modelling discontinuous trapped concentration profiles

[Task 14](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task14.ipynb): Non-cartesian meshes

[Task 15](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task15.ipynb): Kinetic surface model

[Task 16](https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task16.ipynb): Custom classes



## Getting started

### A. Binder (recommended)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/festim-dev/FESTIM-workshop/main)

[Run this workshop](https://mybinder.org/v2/gh/festim-dev/FESTIM-workshop/main) in BinderHub

### B. Codespaces

You can [create a Codespace](https://github.com/codespaces/new?machine=standardLinux32gb&repo=520445592&ref=main&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=WestEurope) based on this repo

### C. Local install

1. Clone this repo

```
git clone https://github.com/festim-dev/FESTIM-workshop
```
2. Create Conda environment (requires conda)

```
conda env create -f environment.yml
```

3. You should then be able to execute the notebooks with the ``festim-workshop`` environment