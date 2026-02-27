# Installation

## Prerequisites

- 🐍 **Python** 3.9 – 3.11
- ⚙️ **MPI compiler** (`mpicc`) for parallel execution

## Clone the repo

```bash
git clone https://github.com/Yi5817/Genarris.git
cd Genarris
git submodule update --init --recursive
```

## Create a virtual environment

`````{tab-set}

````{tab-item} venv

```bash
python3.11 -m venv gnrs_env
source gnrs_env/bin/activate
```
````

````{tab-item} virtualenv
```bash
virtualenv -p python3.11 gnrs_env
source gnrs_env/bin/activate
```
````

````{tab-item} conda
```bash
conda create -n gnrs_env python=3.11
conda activate gnrs_env
```
````

`````

## Install MPI Support

Install `mpi4py` with the system MPI compiler:

```bash
MPICC=$(which mpicc) pip install mpi4py==3.1.5
```

:::{note}
On HPC systems, the C MPI compiler may differ (e.g., `cc` on Cray systems).
Refer to the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html)
or your system administrator.
:::

## Install the package

```bash
pip install -e .
```

:::{note}
`mpicc` is used to build the C extensions. To use a specific MPI compiler,
modify the `mpi_compiler` variable in `setup.py`.
:::

## Optional Energy Calculators

Genarris supports various energy calculators through the
[ASE Calculator interface](https://ase-lib.org/).

### Machine Learning Interatomic Potentials (MLIPs)

`````{tab-set}

````{tab-item} UMA
```bash
pip install -e .[uma]
```

UMA requires a HuggingFace account with access to the
[UMA model repository](https://huggingface.co/facebook/UMA).
````

````{tab-item} MACE-OFF
```bash
pip install -e .[mace]
```
````

````{tab-item} AIMNet2
```bash
pip install git+https://github.com/isayevlab/aimnetcentral.git
```
````

`````

### DFT and Semi-Empirical

| Calculator | Type | Notes |
|:-----------|:-----|:------|
| [DFTB+](https://dftbplus.org/) | Semi-Empirical | Install separately; provide path in config |
| [FHI-aims](https://fhi-aims.org/) | DFT | Licensed software; provide binary path in config |
| [VASP](https://www.vasp.at/) | DFT | Licensed software; provide binary path in config |
