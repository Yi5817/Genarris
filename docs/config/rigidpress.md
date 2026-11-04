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
  value in `[crystal_generation]` (tighter packing).

`maxiter` : `int` | default = `400`.
: Maximum optimization iterations.

`natural_cutoff_mult` : `float` | default = `1.1`.
: Bond identification multiplier. Same as `crystal_generation`.

`debug_flag` : `bool` | default = `False`.
: Write intermediate `geometry.in` files for debugging.

---

## `symm_rigid_press`

Symmetry-preserving rigid press geometry optimization. Minimizes the cell
volume plus a regularized contact energy with
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
and an analytic gradient.

For multi-component crystals, every molecule of the asymmetric unit is a
separate rigid body; the stoichiometry is taken from `[crystal_generation]`.
If molecules of the asymmetric unit already overlap at the `sr` cutoff, the
whole asymmetric unit is pressed as one rigid body. A molecule on a special
Wyckoff position keeps its site symmetry: its centre moves only along the
site and it rotates only about an axis compatible with the site.

```ini
[symm_rigid_press]
sr                  = 0.85
method              = BFGS
int_scale           = 0.1
natural_cutoff_mult = 1.1
tol                 = 0.01
maxiter             = 5000
debug_flag          = False
symprec             = 0.001
batch_size          = 0
```

`sr` : `float` | default = `0.85`.
: Specific radius proportion for the cutoff matrix.

`method` : `str` | default = `BFGS`.
: Scipy local optimizer. Only `BFGS` is tested; `L-BFGS-B` reports success
  at the starting structure and is rejected.

`int_scale` : `float` | default = `0.1`.
: Scale factor for the regularized interatomic contact interaction.

`natural_cutoff_mult` : `float` | default = `1.1`.
: Bond identification multiplier.

`tol` : `float` | default = `0.01`.
: Convergence tolerance passed to `scipy.optimize.minimize` (for `BFGS` the
  maximum gradient component, in $\mathrm{\AA}^2$ for lattice parameters and
  $\mathrm{\AA}^3$ for molecule parameters).

`maxiter` : `int` | default = `5000`.
: Maximum optimization iterations.

`debug_flag` : `bool` | default = `False`.
: Write intermediate geometry files for debugging.

`symprec` : `float` | default = `0.001`.
: spglib tolerance in $\mathrm{\AA}$ used to detect the space group and the
  Wyckoff sites that the optimization preserves.

`batch_size` : `int` | default = `0`.
: Number of crystals optimized at once with the batched torch backend. Rank 0
  collects the structures of all ranks and optimizes them on its GPU, which
  is required; a few hundred saturate one GPU. `0` optimizes one crystal at
  a time with scipy on every rank. The two backends take different paths and
  can end in different local minima.
