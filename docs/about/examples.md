# Examples

This page provides example configurations and workflows for common CSP scenarios.

## Basic Generation + Rigid Press

The simplest workflow generates random crystal structures and optimizes them
with symmetry-preserving rigid press:

```ini
[master]
name                        = my_molecule
molecule_path               = ["molecule.xyz"]
Z                           = 4
log_level                   = info

[workflow]
tasks = ['generation', 'symm_rigid_press']

[generation]
num_structures_per_spg      = 4000
sr                          = 0.95
max_attempts_per_spg        = 100000000
tol                         = 0.01
unit_cell_volume_mean       = predict
volume_mult                 = 1.5
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
```

```bash
mpirun -np 16 gnrs --config ui.conf
```

## Full CSP Workflow
