FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

RUN conda install -c conda-forge mamba
RUN mamba install -c conda-forge fenics

RUN pip install festim
RUN pip install meshio[all]