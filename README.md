# FESTIM-workshop

[![Test](https://github.com/RemDelaporteMathurin/FESTIM-workshop/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/RemDelaporteMathurin/FESTIM-workshop/actions/workflows/test_notebooks.yml)

## Contents

[Task 1](https://github.com/jhdark/FESTIM-workshop/tree/KF_workshop/tasks/task1.ipynb): Simple hydrogen transport simulation

[Task 2](https://github.com/jhdark/FESTIM-workshop/tree/KF_workshop/tasks/task2.ipynb): Gas driven permeation

[Task 3](https://github.com/jhdark/FESTIM-workshop/tree/KF_workshop/tasks/task3.ipynb): Multi-material modelling: LiPb pipe flow

[Task 4](https://github.com/jhdark/FESTIM-workshop/tree/KF_workshop/tasks/task4.ipynb): Simulation of a Thermo-Desorption experiment


## Getting started

### A. Codespaces (recommended)

You can [create a Codespace](https://github.com/codespaces/new?machine=standardLinux32gb&repo=520445592&ref=main&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=WestEurope) based on this repo

Once in the codespace run the following commands:
```
conda create -n festim-env
source activate festim-env
bash setup.sh
```

### B. Local install

1. Clone this repo

```
git clone https://github.com/jhdark/FESTIM-workshop/tree/KF_workshop
```
2. [Install FESTIM](https://festim.readthedocs.io/en/latest/getting_started.html)

3. Install dependencies

```
pip install -r requirements.txt
```
