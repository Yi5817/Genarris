# Rigid Press

## `rigid_press`

Original rigid press geometry optimizer (C implementation). May break crystal symmetry
during optimization.

```ini
[rigid_press]
sr                  = 0.85
maxiter             = 400
natural_cutoff_mult = 1.1
debug_flag          = False
```

`sr` : `float` | default = `0.85`.
: Specific radius proportion for the cutoff matrix. Should be smaller than the
  value in `[generation]` (tighter packing).

`maxiter` : `int` | default = `400`.
: Maximum optimization iterations.

`natural_cutoff_mult` : `float` | default = `1.1`.
: Bond identification multiplier. Same as `generation`.

`debug_flag` : `bool` | default = `False`.
: Write intermediate `geometry.in` files for debugging.

---

## `symm_rigid_press`

Symmetry-preserving rigid press geometry optimization. Uses
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

```ini
[symm_rigid_press]
sr                  = 0.85
method              = BFGS
int_scale           = 0.1
natural_cutoff_mult = 1.1
tol                 = 0.0001
vol_tol             = 10
maxiter             = 5000
debug_flag          = False
```

`sr` : `float` | default = `0.85`.
: Specific radius proportion for the cutoff matrix.

`method` : `str` | default = `BFGS`.
: Scipy local optimizer (e.g. `BFGS`, `L-BFGS-B`, `CG`).

```{note}
Check out the [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) for more details.
```

`int_scale` : `float` | default = `0.1`.
: Scale factor for the regularized interatomic contact interaction.

`natural_cutoff_mult` : `float` | default = `1.1`.
: Bond identification multiplier.

`tol` : `float` | default = `0.01`.
: Convergence tolerance for the optimizer.

`vol_tol` : `float` | default = `10`.
: Volume tolerance in $\mathrm{\AA}^3$.

`maxiter` : `int` | default = `5000`.
: Maximum optimization iterations.

`debug_flag` : `bool` | default = `False`.
: Write intermediate geometry files for debugging.
