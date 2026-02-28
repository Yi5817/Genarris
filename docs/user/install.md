# Installation

## Prerequisites

- 🐍 **Python** 3.10 – 3.11
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

## Install Build Dependencies

Install the build dependencies **before** installing the package, so that
`mpi4py` is compiled against the correct MPI compiler on your system:

```bash
pip install "setuptools>=61.0" wheel "swig>=4.1,<4.3" Cython "numpy>=2.0,<2.3"
MPICC=$(which mpicc) pip install mpi4py --no-cache-dir
```

:::{note}
On HPC systems, the C MPI compiler may differ (e.g., `cc` on Cray systems).
Refer to the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html)
or your system administrator.
:::

## Install the package

Use `--no-build-isolation` so the build uses the `mpi4py` you just installed
(instead of re-installing it in an isolated environment):

```bash
pip install -e . --no-build-isolation
```

:::{note}
**MPI compiler:** `mpicc` is used to build the C extensions. To use a
different MPI compiler, modify the `MPICC` variable in `setup.py`.

**BLAS/LAPACK:** The `rigid_press` extension links against BLAS and LAPACK
(`-llapack -lblas` by default). On HPC systems the library names or paths may
differ — edit the `libraries` and `library_dirs` fields of the `rigid_press`
Extension in `setup.py` before installing.

**Example — TACC Vista (aarch64, NVPL):**
Vista uses [NVIDIA Performance Libraries (NVPL)](https://docs.tacc.utexas.edu/hpc/vista/#compiler-examples)
instead of the standard BLAS/LAPACK. After loading the `nvpl` module
(`module load nvpl`), update `setup.py`:
```python
rigid_press = Extension(
    "gnrs.cgenarris.src.rpack.rigid_press._rigid_press",
    include_dirs=include_rigid_press,
    sources=sources_rigid_press,
    extra_compile_args=["-std=gnu99", "-O3"],
    libraries=["nvpl_blas_lp64_gomp", "nvpl_lapack_lp64_gomp"],
    library_dirs=[os.path.join(os.environ.get("TACC_NVPL_DIR"), "lib")],
    swig_opts=["-I./gnrs/cgenarris/src/rpack/rigid_press", "-I./gnrs/cgenarris/src/spglib_src"],
)
```
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
