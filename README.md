# PNFDM Conda Environment Setup

This document provides instructions for setting up the `pnfdm` Conda environment on a new machine using the exported environment configuration file.

## Prerequisites

Before setting up the environment, make sure you have the following installed:

- [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Steps to Setup Environment

### 1. Clone or Download the Repository

Clone or download the project repository that includes the environment configuration file (`pnfdm_environment.yml`).

```bash
git clone <https://github.com/zzfu-buaa/PNFDM.git>
cd <PNFDM>
```

### 2. Create the Conda Environment
Run the following command to create a new Conda environment using the exported YAML configuration file:
```bash
conda env create -f pnfdm_environment.yml
```
This will create a new Conda environment named pnfdm with all the dependencies specified in the pnfdm_environment.yml file.

### 3. Activate the Environment
After the environment has been created, activate it by running:
```bash
conda activate pnfdm
```
You should now be in the pnfdm environment, and you can start working with the project's dependencies.

### 4. Verify the Environment
To verify that the environment has been correctly set up, check the installed packages and versions:
```bash
conda list
This will show you all the packages installed in the pnfdm environment and their versions.
```
### 5. Deactivate the Environment
Once you are done working in the environment, you can deactivate it using the following command:
```bash
conda deactivate
```
This will return you to your base Conda environment or system Python.

Updating the Environment

If you make changes to the environment (e.g., install new packages), you can update the pnfdm_environment.yml file by running:
```bash
conda env export --name pnfdm > pnfdm_environment.yml
```
This will regenerate the YAML file with the new dependencies, which can be shared and used to recreate the environment.