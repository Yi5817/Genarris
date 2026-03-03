# Energy Calculation

Each energy calculator has its own config section. Available calculators: `maceoff`, `uma`, `aimnet`, `aims`, `vasp`, `dftb`.

---

## `maceoff`

[MACE-OFF](https://github.com/ACEsuit/mace) machine-learned interatomic
potential. Requires GPU.

```ini
[maceoff]
model_size  = large
save_flag   = True
```

`model_size` : `str` | default = `large`.
: Model size (`small`, `medium`, `large`).

`save_flag` : `bool` | default = `True`.
: Checkpoint after each structure.

---

## `uma`

[UMA](https://github.com/facebookresearch/fairchem) (Universal Model for
Atoms) from Meta FAIR. Requires GPU.

```ini
[uma]
model_name  = uma-s-1p1
task_name   = omc
```

`model_name` : `str` | default = `uma-s-1p1`.
: Pretrained model name. See [fairchem docs](https://fair-chem.github.io/quickstart/).

`task_name` : `str` | default = `omc`.
: The task specifies the level of theory/DFT calculations to emulate. See [fairchem docs](https://fair-chem.github.io/quickstart/#available-tasks).

---

## `aimnet`

[AIMNet2](https://github.com/isayevlab/aimnetcentral) neural network potential.
Requires GPU.

```ini
[aimnet]
model = aimnet2
```

`model` : `str` | default = `aimnet2`.
: AIMNet2 model name or model path.

---

## DFT execution modes

DFT calculators (`aims`, `vasp`) support two execution modes via the
`dft_mode` option:

`parallel` (default)
: Every Genarris MPI rank launches its own DFT subprocess.  This is the
  original behavior, suitable when each rank has dedicated cores for its
  DFT job (e.g. via `use_slurm` host pinning).

`serial`
: Structures from **all** ranks are gathered to rank 0 via MPI.  Rank 0
  runs each DFT calculation sequentially, giving the DFT binary the
  full SLURM / MPI allocation (e.g. `srun -n 256 aims.x`).  Results are
  broadcast back so every rank has the computed energies.  This avoids
  nested-MPI conflicts and is recommended when you have relatively few
  structures but need maximum parallelism per DFT job.

  Works with any number of Genarris ranks -- use many ranks for earlier
  steps (generation, clustering) and the same job transitions seamlessly
  into serial DFT when that task is reached.

:::{dropdown} SLURM script + config for ``dft_mode = serial`` (recommended)
:icon: terminal

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=256
#SBATCH --time=48:00:00

# 256 ranks for generation/clustering/descriptor steps.
# During DFT tasks, all structures are gathered to rank 0, which
# calls "srun -n 256 aims.x" for each structure one at a time.
mpirun -np 256 gnrs -c ui.conf
```

```ini
[aims]
command               = /path/to/aims.x
species_dir           = /path/to/species_defaults/defaults_2020/light
energy_settings_path  = ./aims_settings.json
num_cores             = 256
mpi_launcher          = srun
dft_mode              = serial
```

With this setup, during DFT steps rank 0 calls
`srun -n 256 /path/to/aims.x` for each structure sequentially.
Other Genarris ranks wait at the MPI gather/broadcast.  `num_cores`
can be set independently of the Genarris rank count -- e.g. 128 Genarris
ranks for generation but `num_cores = 256` to give each DFT job more
cores than Genarris ranks.
:::

:::{dropdown} SLURM script + config for ``dft_mode = parallel``
:icon: terminal

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=256
#SBATCH --time=48:00:00

# 4 Genarris ranks, each launches its own DFT subprocess with 64 cores.
# Total: 4 parallel DFT jobs * 64 cores = 256 cores.
mpirun -np 4 gnrs -c ui.conf
```

```ini
[aims]
command               = /path/to/aims.x
species_dir           = /path/to/species_defaults/defaults_2020/light
energy_settings_path  = ./aims_settings.json
num_cores             = 64
mpi_launcher          = mpirun
dft_mode              = parallel
use_slurm             = True
```

Each rank runs its own DFT subprocess pinned to a specific SLURM host.
Use this when you have many structures and want parallel DFT jobs.
:::

---

## `aims`

[FHI-aims](https://fhi-aims.org/) all-electron DFT via the ASE calculator.

```ini
[aims]
command               = /path/to/aims.x
species_dir           = /path/to/species_defaults/defaults_2020/light
energy_settings_path  = ./aims_settings.json
num_cores             = 128
mpi_launcher          = mpirun
dft_mode              = parallel
use_slurm             = False
```

`command` : `str`.
: Path to the FHI-aims binary. Do **not** include `mpirun`/`srun` here;
  the MPI launcher is controlled by `mpi_launcher`.

`species_dir` : `str`.
: Path to the FHI-aims `species_defaults` directory.

`energy_settings_path` : `str`.
: Path to a JSON file with FHI-aims settings
  (passed to [`ase.calculators.aims.Aims`](https://ase-lib.org/ase/calculators/FHI-aims.html)).

:::{dropdown} Example `aims_settings.json`
:icon: file-code

```json
{
  "spin": "none",
  "relativistic": "atomic_zora scalar",
  "charge": 0,
  "occupation_type": "gaussian 0.01",
  "mixer": "pulay",
  "n_max_pulay": 8,
  "charge_mix_param": 0.2,
  "sc_accuracy_etot": 1e-06,
  "sc_accuracy_rho": 1e-05,
  "sc_accuracy_forces": 0.0001,
  "sc_accuracy_eev": 0.001,
  "sc_iter_limit": 10000,
  "KS_method": "parallel",
  "empty_states": 6,
  "basis_threshold": 1e-05,
  "many_body_dispersion": " ",
  "compute_forces": ".true."
}
```
:::

`num_cores` : `int`.
: CPU cores per DFT process. Required when `mpi_launcher` is not `none`.

`mpi_launcher` : `str` | default = `mpirun`.
: MPI launcher to use for DFT runs. One of:
  - `mpirun` -- launch with `mpirun -np {num_cores}`.
  - `srun` -- launch with `srun -n {num_cores}`.
  - `ibrun` -- launch with `ibrun`.
  - `none` -- run the binary directly with no MPI wrapper. Use this when
    the job is already running inside an MPI allocation (e.g. the SLURM
    job script already uses `srun`).

`dft_mode` : `str` | default = `parallel`.
: DFT execution mode. One of:
  - `parallel` -- every Genarris MPI rank runs its own DFT subprocess
    (original behavior).
  - `serial` -- structures are gathered to rank 0, which runs DFT
    sequentially.  Results are broadcast back.  The DFT binary gets the
    full allocation.  Works with any number of Genarris ranks (e.g.
    `mpirun -np 128 gnrs ...` for generation, then serial DFT).

`use_slurm` : `bool` | default = `False`.
: When `mpi_launcher` is `mpirun` and `dft_mode` is `parallel`, parse
  SLURM node list and pin each rank to a specific host. Ignored when
  `mpi_launcher` is `srun` or `none`, or when `dft_mode` is `serial`.

---

## `vasp`

[VASP](https://www.vasp.at/) plane-wave DFT via the ASE calculator.

```ini
[vasp]
command               = /path/to/vasp_std
energy_settings_path  = ./vasp_settings.json
num_cores             = 128
mpi_launcher          = mpirun
dft_mode              = parallel
use_slurm             = False
```

`command` : `str`.
: Path to the VASP binary. Do **not** include `mpirun`/`srun` here;
  the MPI launcher is controlled by `mpi_launcher`.

`energy_settings_path` : `str`.
: Path to a JSON file with VASP settings
  (passed to [`ase.calculators.vasp.Vasp`](https://ase-lib.org/ase/calculators/vasp.html#id2)).

`num_cores` : `int`.
: CPU cores per DFT process. Required when `mpi_launcher` is not `none`.

`mpi_launcher` : `str` | default = `mpirun`.
: MPI launcher for DFT runs. One of `mpirun`, `srun`, `ibrun`, or `none`.
  See the `aims` section above for details.

`dft_mode` : `str` | default = `parallel`.
: DFT execution mode (`serial` or `parallel`).
  See the `aims` section above for details.

`use_slurm` : `bool` | default = `False`.
: When `mpi_launcher` is `mpirun` and `dft_mode` is `parallel`, pin
  each rank to a SLURM host. Ignored for `srun`, `none`, or `serial`.

---

## `dftb`

[DFTB+](https://dftbplus.org/) density functional tight-binding via the ASE
calculator.

```ini
[dftb]
command               = dftb+ > dftb.out
sk_files              = ./3ob-3-1/
energy_settings_path  = ./dftb_settings.json
```

`command` : `str`.
: Shell command to run DFTB+.

`sk_files` : `str`.
: Path to the Slater-Koster parameter files.

`energy_settings_path` : `str`.
: Path to a JSON file with DFTB+ settings
  (passed to [`ase.calculators.dftb.Dftb`](https://ase-lib.org/ase/calculators/dftb.html#dftb-calculator-class)).

:::{dropdown} Example `dftb_settings.json`
:icon: file-code

```json
{
    "Hamiltonian_MaxAngularMomentum_": "",
    "Hamiltonian_MaxAngularMomentum_C": "\"p\"",
    "Hamiltonian_MaxAngularMomentum_H": "\"s\"",
    "Hamiltonian_MaxAngularMomentum_O": "\"p\"",
    "Hamiltonian_MaxAngularMomentum_N": "\"p\"",
    "Hamiltonian_SCC": "Yes",
    "Hamiltonian_ThirdOrderFull": "Yes",
    "kpts": "[3, 3, 3, 'gamma']",
    "Hamiltonian_HubbardDerivs_": "",
    "Hamiltonian_HubbardDerivs_C": "-0.1492",
    "Hamiltonian_HubbardDerivs_H": "-0.1857",
    "Hamiltonian_HubbardDerivs_O": "-0.1575",
    "Hamiltonian_HubbardDerivs_N": "-0.1535",
    "Hamiltonian_HCorrection_": "Damping",
    "Hamiltonian_HCorrection_Exponent": "4.000",
    "Hamiltonian_Dispersion_": "DftD3",
    "Hamiltonian_Dispersion_Damping_": "BeckeJohnson",
    "Hamiltonian_Dispersion_Damping_a1": "0.746",
    "Hamiltonian_Dispersion_Damping_a2": "4.191",
    "Hamiltonian_Dispersion_s6": "1.0",
    "Hamiltonian_Dispersion_s8": "3.209"
}
```
:::
