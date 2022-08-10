FROM continuumio/miniconda3

RUN conda install -c conda-forge fenics

RUN pip install festim==0.10.2 matplotlib meshio[all] ipykernel h-transport-materials==0.4.0