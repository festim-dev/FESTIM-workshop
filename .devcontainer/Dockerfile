FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

COPY environment.yml* .

RUN /opt/conda/bin/conda env create -f environment.yml

# RUN apt-get update && apt-get install -y libgl1
