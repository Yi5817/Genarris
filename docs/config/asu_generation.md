# ASU Generation

Random asymmetric unit (ASU) generation for multi-component crystals
(co-crystals, salts, solvates, Z' > 1) via `cgenarris`.

An ASU is a small cluster of molecules, for example one molecule A and one
molecule B of a 1:1 co-crystal. The task places the molecules at random and
keeps a cluster when the molecules are in contact but do not overlap.

```{note}
`asu_generation` writes non-periodic structures. Crystal generation from the
ASUs is not available yet. For single-component crystals, use
{doc}`crystal_generation`.
```

## Example

A 1:1 co-crystal needs one geometry file per molecule:

```ini
[master]
name                 = cocrystal
molecule_path        = ["molecule_a.xyz", "molecule_b.xyz"]

[workflow]
tasks                = ['asu_generation']

[asu_generation]
stoichiometry        = [1, 1]
num_asus             = 15000
```

Run it:

```bash
mpirun -np 8 gnrs -c ui.conf
```

The ASUs are in `structures/asu_generation/structures.json`. Load them, or
export them to one `.xyz` file for a viewer:

```python
import ase.io
from ase.io.jsonio import read_json

asus = read_json("structures/asu_generation/structures.json")
ase.io.write("asus.xyz", list(asus.values()))
```

In each ASU the atoms keep the input order: all atoms of `molecule_a`, then
all atoms of `molecule_b`.

## Options

```ini
[asu_generation]
stoichiometry        = [1, 1]
num_asus             = 15000
sr_min               = 0.75
sr_max               = 1.3
max_attempts_per_asu = 100000
seed                 = 42
```

`stoichiometry` : `list[int]`.
: Copies of each molecule per ASU, in the order of `molecule_path`.
  `[1, 1]` is one molecule A and one molecule B. `[1, 2]` is one A and two B.
  `[2]` with one molecule file is a dimer (Z' = 2). The sum must be at least 2.

`num_asus` : `int`.
: Total number of ASUs to generate.

`sr_min` : `float` | default = `0.75`.
: Lower bound of the contact window. `sr` is the closest distance between
  atoms of two different molecules, divided by the sum of their van der Waals
  radii. An ASU with `sr` below `sr_min` has overlapping molecules and is
  rejected.

`sr_max` : `float` | default = `1.3`.
: Upper bound of the contact window. An ASU with `sr` above `sr_max` has
  molecules that are too far apart and is rejected.

`max_attempts_per_asu` : `int` | default = `100000`.
: Maximum random placements per ASU. An MPI process that reaches this limit
  stops early, and Genarris prints a warning. If the pool holds fewer than
  `num_asus` ASUs, increase this value or widen the `sr` window.

`seed` : `int` | default = `42`.
: Random seed for reproducibility. `0` uses a time-based seed.
