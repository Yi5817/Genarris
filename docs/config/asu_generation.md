# ASU Generation

Random asymmetric unit (ASU) generation for multi-component crystals
(co-crystals, salts, solvates, Z' > 1) via `cgenarris`.

An ASU is a small cluster of molecules, for example one molecule A and one
molecule B of a 1:1 co-crystal. The task places the molecules at random and
keeps a cluster when the molecules are in contact but do not overlap.

```{note}
`asu_generation` writes non-periodic structures. To build crystals from them,
put `crystal_generation` after it, see [Crystals from ASUs](#crystals-from-asus).
For single-component crystals, use {doc}`crystal_generation` alone.
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

```{warning}
If no ASU is accepted, the task stops with an error. Structures from an
earlier run stay on disk; do not use them.
```

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
  This release supports only two molecules with `[1, 1]`: one molecule A and
  one molecule B.

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

`seed` : `int` | default = the `--seed` value of the run (`42`).
: Random seed of this task. `0` uses a time-based seed.

## Crystals from ASUs

A `crystal_generation` task after `asu_generation` generates crystals from
every ASU of the pool. It places the ASU as one rigid unit, so `z` is the
number of ASUs per cell, not the number of molecules.

```ini
[master]
name                   = cocrystal
molecule_path          = ["molecule_a.xyz", "molecule_b.xyz"]
z                      = 2

[workflow]
tasks                  = ['asu_generation', 'crystal_generation']

[asu_generation]
stoichiometry          = [1, 1]
num_asus               = 5

[crystal_generation]
stoichiometry          = [1, 1]
num_structures_per_spg = 2000
```

```{warning}
Every ASU gets `num_structures_per_spg` crystals per space group. Keep
`num_asus` small.
```

Set `stoichiometry` in both sections. The crystals of all ASUs are in
`structures/crystal_generation/structures.json`. The run of each ASU is in
`tmp/crystal_generation/<ASU name>/`. Each ASU gets its own seed, derived from
`seed` with `numpy.random.SeedSequence`.
