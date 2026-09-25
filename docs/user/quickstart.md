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
[`ase.io.read()`](https://docs.ase-lib.org/ase/io/io.html#ase.io.read)
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
| `--restart` | Resume the previous run in the current directory, skipping completed tasks and structures | — |
| `--overwrite` | Start over in a directory that contains a previous run, discarding its progress | — |

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
├── restart.json
└── Genarris.log
```

## Restarting an Interrupted Run

Genarris records its progress so that a run killed by a walltime limit or a
node failure can be resumed instead of started over. From the same directory,
rerun the same command with `--restart`:

```bash
mpirun -np <num_processes> gnrs -c ui.conf --restart
```

On startup Genarris lists which tasks are already completed and which task it
resumes from. Progress is kept at two levels:

- **Completed tasks** are recorded in `restart.json` after each task finishes.
  They are skipped on restart and their results in `structures/` are reused.
- **Completed structures** of the task that was running are logged in
  `tmp/<task>/rank_*/*.ckpt` as they finish. On restart only the remaining
  structures are computed.

Things that are safe to change between the original run and the restart:

- **Number of MPI processes.** Checkpointed structures are rebalanced across
  the new process count.
- **Run directory location.** If the directory was moved or renamed, saved
  paths are remapped automatically.
- **Config settings.** The current config file takes precedence; every setting
  that differs from the original run is listed at startup.
- **Tasks appended to the end** of `[workflow] tasks`. Inserting, removing or
  reordering tasks *before* a completed one is refused, because completed
  tasks are matched by their position in the list. Appending a second task of
  a type that is already in the list is refused too, because duplicate tasks
  are renumbered (`dedup_1`, `dedup_2`) and the completed one would no longer
  be found; start over with `--overwrite` in that case.

If a restart cannot proceed, Genarris stops with a message explaining why
(for example, no `restart.json` in the current directory, the structure file
of the last completed task was deleted, or the task list changed). A checkpoint
line cut short when the job was killed is skipped with a warning and that
structure is recomputed.

### Starting over

Running *without* `--restart` in a directory that already contains
`restart.json` is refused, so a finished run cannot be overwritten by accident.
Pass `--overwrite` to discard the previous progress and start fresh:

```bash
mpirun -np <num_processes> gnrs -c ui.conf --overwrite
```

:::{note}
Runs made with older Genarris releases keep the restart file at
`tmp/restart.json`. Move it to the run directory to resume them.
:::

Structures are stored as JSON ASE Atoms objects. Load them with:

```python
import json
from ase.io.jsonio import read_json

xtals = read_json("structures/symm_rigid_press/structures.json")
```
