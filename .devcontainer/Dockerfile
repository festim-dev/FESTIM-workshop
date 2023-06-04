FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

RUN conda install -c conda-forge mamba
RUN mamba install -c conda-forge fenics

RUN apt-get update && apt-get install -y libgl1

RUN pip install festim matplotlib meshio[all] ipykernel h-transport-materials==0.4.0
