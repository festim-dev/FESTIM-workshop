FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

RUN apt-get update && apt-get install -y libgl1

RUN conda install conda==23.5.2
RUN conda install -c conda-forge fenics

RUN pip install festim==0.11.1 matplotlib meshio[all] ipykernel numpy==1.24