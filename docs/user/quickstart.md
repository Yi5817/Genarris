# Quick Start

This guide walks you through running your first crystal structure prediction
workflow with Genarris.

## Overview

Genarris uses a [configuration file](https://docs.python.org/3/library/configparser.html) 
to control each step of the CSP pipeline. A typical workflow consists of:

1. **Structure Generation** – Random crystal structures across space groups
2. **Rigid Press** – Geometry optimization to improve close-packed structures
3. **Energy Evaluation** – Compute energies and relax geometries with MLIPs or DFT
4. **Descriptor Computation** – Calculate fingerprints (ACSF)
5. **Clustering & Selection** – Group similar structures and select representatives

## Step 1: Prepare Conformer

Provide conformer geometry in any format supported by
[`ase.io.read()`](https://ase-lib.org/ase/io/io.html#ase.io.read)
(e.g., `.xyz`, `.mol`, `.in`).

## Step 2: Create Configuration File

Create a file named `ui.conf`:

```ini
[master]
name                        = aspirin
molecule_path               = ["aspirin.xyz"]
Z                           = 4
log_level                   = info

[workflow]
tasks                       = ['generation', 'symm_rigid_press']

[generation]
num_structures_per_spg      = 4000
sr                          = 0.95
max_attempts_per_spg        = 100000000
tol                         = 0.01
ucv_mean                    = predict
ucv_mult                    = 1.5
max_attempts_per_volume     = 10000000
spg_distribution_type       = standard
generation_type             = crystal
natural_cutoff_mult         = 1.2

[symm_rigid_press]
sr                          = 0.85
method                      = BFGS
tol                         = 0.01
natural_cutoff_mult         = 1.2
debug_flag                  = False
maxiter                     = 5000

[experimental_structure]
path = ""
```

### Key Parameters

| Parameter | Section | Type | Description |
|:----------|:--------|:-----|:------------|
| `name` | `[master]` | `str` | Project name, used for output directories and logs |
| `molecule_path` | `[master]` | `list[str]` | Paths to conformer geometry files |
| `Z` | `[master]` | `int` | Number of molecules per unit cell |
| `tasks` | `[workflow]` | `list[str]` | Ordered list of pipeline tasks to execute |
| `num_structures_per_spg` | `[generation]` | `int` | Structures to generate per space group |

For the full list of options (generation, rigid press, energy calculators,
clustering, etc.), see the {doc}`/config/index` reference.

## Step 3: Run Genarris

```bash
mpirun -np <num_processes> gnrs -c ui.conf
```

**CLI flags:**

| Flag | Description | Default |
|:-----|:------------|:--------|
| `-c`, `--config` | Path to the configuration file (required) | — |
| `-d`, `--seed` | Random seed for reproducibility | `42` |
| `--restart` | Restart from a previous run using the same config file | — |

For example, to run on 8 MPI processes with a specific seed:

```bash
mpirun -np 8 gnrs -c ui.conf -d 42
```

:::{tip}
Use as many MPI processes as available CPU cores. For GPU-accelerated energy
calculators, Genarris automatically manages the GPU worker/feeder pattern.
:::

## Step 4: Results

After running, Genarris creates the following directory structure:

```text
working_directory/
├── structures/
│   ├── generation/
│   │   └── structures.json
│   └── symm_rigid_press/
│       └── structures.json
├── tmp/
│   ├── generation/
│   └── symm_rigid_press/
└── Genarris.log
```

Structures are stored as JSON ASE Atoms objects. Load them with:

```python
import json
from ase.io.jsonio import read_json

xtals = read_json("structures/symm_rigid_press/structures.json")
```
