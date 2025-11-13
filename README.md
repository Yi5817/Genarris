<h1 align="center"> Genarris </h1>

<p align="center">
  <img width="500" src="assets/Genarris_logo.png" alt="Genarris Logo">
</p>

<p align="center">
  <a href="https://github.com/Yi5817/Genarris/releases"><img src="https://img.shields.io/badge/version-3.0.0-green" alt="Version 3.0.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.8+"></a>
  <a href="https://github.com/Yi5817/Genarris/actions/workflows/lint.yml"><img src="https://github.com/Yi5817/Genarris/actions/workflows/lint.yml/badge.svg" alt="Code Style"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue" alt="License"></a>
  <a href="https://doi.org/10.1021/acs.jctc.5c01080"><img src="https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.5c01080-blue" alt="DOI"></a>
</p>


Genarris `gnrs` is a random molecular crystal structure generator and a computational workflow for molecular crystal structure prediction (CSP).

## Installation

Clone the repository:

  ```bash
  git clone https://github.com/Yi5817/Genarris.git
  cd Genarris
  git submodule update --init --recursive
  ```

Create and activate the virtual enviornment using your favorite venv tool:

```bash
virtualenv -p python3.11 gnrs_env
source gnrs_env/bin/activate
```

### Install MPI Support

Install `mpi4py` with the MPI compiler:

```bash
MPICC=$(which mpicc) pip install mpi4py==3.1.5
```

> [!NOTE]
> On high-performance computing machine, C MPI compiler may differ. Please refer to the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html) or contact the system administrator.

### Install `gnrs` using pip

   ```bash
   pip install -e .
   ```

   > [!NOTE]
   > `mpicc` is used to build the C extensions. To use a specific MPI compiler, modify the `mpi_compiler` variable in the [setup.py](./setup.py) file.

### Optional Energy Calculators

Genarris supports various energy calculators through the [ASE Calculator](https://ase-lib.org/). These enable energy evaluation and geometry relaxation with machine learning interatomic potentials (MLIPs), force field/semi-empirical methods, and DFT packages.

> [!TIP]
> You can implement additional calculators under [`gnrs/energy/`](./gnrs/energy/).

| Calculator | Type | Install | 
|------------|------|---------|
| [UMA](https://github.com/facebookresearch/fairchem) | MLIP | `pip install -e .[uma]` |
| [MACE-OFF](https://github.com/ACEsuit/mace) | MLIP | `pip install -e .[mace]` |
| [AIMNet2](https://github.com/isayevlab/aimnetcentral) | MLIP | `pip install git+https://github.com/isayevlab/aimnetcentral.git` |
| [DFTB+](https://dftbplus.org/) | Semi-Empirical | — |
| [FHI-aims](https://fhi-aims.org/) | DFT | — |
| [VASP](https://www.vasp.at/) | DFT | — |

> :warning: To access gated UMA models, you need to get a HuggingFace account and request access to the [UMA model repository](https://huggingface.co/facebook/UMA).

## Quick Start

Genarris uses a [configuration file](https://docs.python.org/3/library/configparser.html) to control crystal structure generation and selection.

### Basic Workflow

1. **Create a configuration file** `ui.conf`
   
   Here's an example with key parameters for `generation` and `symm_rigid_press` steps:

   ```ini
   [master]
   name                        = 
   molecule_path               = [""]
   Z                           = 
   log_level                   = info

   [workflow]
   tasks                       = ['generation', 'symm_rigid_press']

   [generation]
   num_structures_per_spg      = 4000
   sr                          = 0.95
   # sr is an alias for specific_radius_proportion
   max_attempts_per_spg        = 100000000
   tol                         = 0.01
   unit_cell_volume_mean       = predict
   volume_mult                 = 1.5 
   max_attempts_per_volume     = 10000000
   spg_distribution_type       = standard
   generation_type             = crystal
   natural_cutoff_mult         = 1.2

   [symm_rigid_press]
   sr                           = 0.85
   method                       = BFGS
   tol                          = 0.01
   natural_cutoff_mult          = 1.2
   debug_flag                   = False
   maxiter                      = 5000

   [experimental_structure]
   path = ""
   ```
2. **Prepare your input molecule geometry file** (any format supported by [`ase.io.read()`](https://ase-lib.org/ase/io/io.html#ase.io.read))
3. **Run Genarris with MPI parallelization**:

   ```bash
   mpirun -np <num_processes> gnrs --config <config_file>
   ```

   For example, to run with 8 processes:
   ```bash
   mpirun -np 8 gnrs --config ui.conf
   ```

## Case Studies

The [cases](./cases) directory contains crystal structure prediction (CSP) results for 6 organic molecules studied in the [Genarris 3.0 paper](https://doi.org/10.1021/acs.jctc.5c01080). Each case includes:
- **Experimental structures**: From [the Cambridge Structural Database (CSD)](https://www.ccdc.cam.ac.uk/solutions/about-the-csd/)
- **Generated structures**: CIF files of crystal structures generated by Genarris and relaxed using MACE-OFF23 and PBE+MBD methods
- **Analysis data**: Total energies from MACE-OFF23 and PBE+MBD methods
  
## Citation

If you use Genarris, please cite our papers:
```bibtex
@article{genarrisv3,
  author = {Yang, Yi and Tom, Rithwik and Wui, Jose AGL and Moussa, Jonathan E and Marom, Noa},
  title = {Genarris 3.0: Generating Close-Packed Molecular Crystal Structures with Rigid Press},
  author={Yang, Yi and Tom, Rithwik and Wui, Jose AGL and Moussa, Jonathan E and Marom, Noa},
  journal={Journal of Chemical Theory and Computation},
  volume = {21},
  number = {21},
  pages = {11318--11332},
  year={2025},
  publisher={ACS Publications}
}

@article{genarrisv2,
  title={Genarris 2.0: A random structure generator for molecular crystals},
  author={Tom, Rithwik and Rose, Timothy and Bier, Imanuel and O’Brien, Harriet and V{\'a}zquez-Mayagoitia, {\'A}lvaro and Marom, Noa},
  journal={Computer Physics Communications},
  volume={250},
  pages={107170},
  year={2020},
  publisher={Elsevier}
}

@article{genarrisv1,
  title={Genarris: Random generation of molecular crystal structures and fast screening with a Harris approximation},
  author={Li, Xiayue and Curtis, Farren S and Rose, Timothy and Schober, Christoph and Vazquez-Mayagoitia, Alvaro and Reuter, Karsten and Oberhofer, Harald and Marom, Noa},
  journal={The Journal of Chemical Physics},
  volume={148},
  number={24},
  year={2018},
  publisher={AIP Publishing}
}

@article{pymove,
  title={Machine learned model for solid form volume estimation based on packing-accessible surface and molecular topological fragments},
  author={Bier, Imanuel and Marom, Noa},
  journal={The Journal of Physical Chemistry A},
  volume={124},
  number={49},
  pages={10330--10345},
  year={2020},
  publisher={ACS Publications}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Feedback and Support

We would appreciate it if you could share any feedback about performance and improvements.

## LICENSE

Genarris is available under the [BSD-3-Clause License](LICENSE).
