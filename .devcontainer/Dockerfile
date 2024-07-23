FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Set the user to vscode
USER vscode

# Copy the environment.yml file into the container
COPY environment.yml* /tmp/

# Create the Conda environment as the vscode user
RUN conda env create -f /tmp/environment.yml

RUN apt-get update && apt-get install -y libgl1
