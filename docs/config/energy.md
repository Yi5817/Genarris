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

## `aims`

[FHI-aims](https://fhi-aims.org/) all-electron DFT via the ASE calculator.

```ini
[aims]
command               = mpirun -np 128 /path/to/aims.x
species_dir           = /path/to/species_defaults/defaults_2020/light
energy_settings_path  = ./aims_settings.json
num_cores             = 128
use_slurm             = False
```

`command` : `str`.
: Command to invoke FHI-aims (including MPI launcher).

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
: CPU cores per FHI-aims process. Used when `use_slurm` is `True`.

`use_slurm` : `bool` | default = `False`.
: Parse SLURM node list and pin each MPI rank to a host.

---

## `vasp`

[VASP](https://www.vasp.at/) plane-wave DFT via the ASE calculator.

```ini
[vasp]
command               = mpirun -np 128 /path/to/vasp_std
energy_settings_path  = ./vasp_settings.json
num_cores             = 128
use_slurm             = False
```

`command` : `str`.
: Command to invoke VASP.

`energy_settings_path` : `str`.
: Path to a JSON file with VASP settings
  (passed to [`ase.calculators.vasp.Vasp`](https://ase-lib.org/ase/calculators/vasp.html#id2)).

`num_cores` : `int`.
: CPU cores per VASP process. Used with `use_slurm`.

`use_slurm` : `bool` | default = `False`.
: Pin ranks to SLURM hosts.

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
