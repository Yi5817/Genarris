# ASE Optimizers

To use BFGS or LBFGS, combine the optimizer and energy method in the task name
(e.g. `lbfgs_maceoff`), and configure each in its own section.

---

## `lbfgs`

Wrapper around
[`ase.optimize.LBFGS`](https://docs.ase-lib.org/ase/optimize.html#ase.optimize.LBFGS).

```ini
[lbfgs]
energy_method   = maceoff
fmax            = 0.01
steps           = 500
cell_opt        = True
fix_sym         = True
```

`energy_method` : `str`.
: Energy calculator to use (e.g. `maceoff`, `uma`, `aims`). Must match a
  configured energy section.

`fmax` : `float`.
: Maximum force convergence criterion in eV/A.

`steps` : `int`.
: Maximum number of optimization steps.

`cell_opt` : `bool`.
: Optimize the unit cell using
  [`FrechetCellFilter`](https://docs.ase-lib.org/ase/filters.html#the-frechetcellfilter-class).

`fix_sym` : `bool`.
: Apply
  [`FixSymmetry`](https://docs.ase-lib.org/ase/constraints.html#the-fixsymmetry-class)
  constraint during optimization.

---

## `bfgs`

Wrapper around
[`ase.optimize.BFGS`](https://docs.ase-lib.org/ase/optimize.html#ase.optimize.BFGS).
Same options as `lbfgs`.

```ini
[bfgs]
energy_method   = maceoff
fmax            = 0.01
steps           = 500
cell_opt        = True
fix_sym         = True
```

`energy_method` : `str`.
: Energy calculator to use.

`fmax` : `float`.
: Maximum force convergence criterion in eV/A.

`steps` : `int`.
: Maximum number of optimization steps.

`cell_opt` : `bool`.
: Optimize the unit cell using
  [`FrechetCellFilter`](https://docs.ase-lib.org/ase/filters.html#the-frechetcellfilter-class).

`fix_sym` : `bool`.
: Apply
  [`FixSymmetry`](https://docs.ase-lib.org/ase/constraints.html#the-fixsymmetry-class)
  constraint during optimization.
