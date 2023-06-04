# FESTIM-workshop

[![Test](https://github.com/RemDelaporteMathurin/FESTIM-workshop/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/RemDelaporteMathurin/FESTIM-workshop/actions/workflows/test_notebooks.yml)

## Contents

[Task 1](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task1.ipynb): Simple hydrogen transport simulation

[Task 2](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task2.ipynb): Simulation of a Thermo-Desorption experiment

[Task 3](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task3.ipynb): Multi-material simulations

[Task 4](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task4.ipynb): Post-processing and visualisation

[Task 5](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task5.ipynb): Advection-diffusion problem

[Task 6](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task6.ipynb): Heat transfer simulation

[Task 7](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task7.ipynb): CAD integration

[Task 8](https://github.com/RemDelaporteMathurin/FESTIM-workshop/blob/main/tasks/task8.ipynb): Integration with the HTM library

## Getting started

### A. Codespaces (recommended)

1. [Sign up to Codespace beta](https://github.com/features/codespaces/signup)

2. [Create a Codespace](https://github.com/codespaces/new?machine=standardLinux32gb&repo=520445592&ref=main&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=WestEurope) based on this repo

### B. Local install

1. Clone this repo

```
git clone https://github.com/RemDelaporteMathurin/FESTIM-workshop
```
2. [Install FESTIM](https://festim.readthedocs.io/en/latest/getting_started.html)

3. Install dependencies

```
pip install matplotlib meshio[all] h-transport-materials
```
