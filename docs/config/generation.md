# Generation

Random crystal structure generation via `cgenarris`.

```ini
[generation]
num_structures_per_spg  = 4000
sr                      = 0.95
max_attempts_per_spg    = 10000000
tol                     = 0.01
ucv_mean                = predict
ucv_mult                = 1.5
ucv_std                 = 0.05
max_attempts_per_volume = 10000
spg_distribution_type   = standard
generation_type         = crystal
natural_cutoff_mult     = 1.1
stoichiometry           = [1]
seed                    = 42
lattice_norm_dev        = 0.4
lattice_angle_std       = 8
```

`num_structures_per_spg` : `int`.
: Number of structures to generate per space group.

`sr` : `float` | default = `0.95`.
: Specific radius proportion. The close-contact cutoff is this fraction of the
  sum of atomic van der Waals radii.
  Also accepted as `specific_radius_proportion`.

`max_attempts_per_spg` : `int` | default = `10000000`.
: Maximum random attempts per space group before giving up.

`tol` : `float` | default = `0.1`.
: Tolerance passed to [spglib](https://spglib.readthedocs.io/) for symmetry
  detection.

`ucv_mean` : `str` | `float` | default = `predict`.
: Mean unit cell volume in $\mathrm{\AA}^3$. Set to `predict` to use the built-in PyMoVE
  ML model. Also accepted as `unit_cell_volume_mean`.

`ucv_std` : `float` | default = `0.05`.
: Standard deviation of the volume distribution, as a fraction of `ucv_mean`.
  Also accepted as `unit_cell_volume_std`.

`ucv_mult` : `float` | default = `1.5`.
: Multiplier for the predicted unit cell volume. `1.0` uses the raw
  prediction. Larger values produce larger cells; subsequent rigid press
  compresses them. Also accepted as `volume_mult`.

`max_attempts_per_volume` : `int` | default = `10000`.
: Maximum attempts per volume point.

`spg_distribution_type` : `str` | `list[int]` | default = `standard`.
: Space group distribution. `standard` uses all compatible with 230 space groups. A list
  of integers (e.g. `[14, 19]`) restricts generation to those space groups only.

`generation_type` : `str` | default = `crystal`.
: Generation mode. Currently only `crystal` is supported.

  ```{note}
  Multi-component crystals are coming soon!
  ```

`natural_cutoff_mult` : `float` | default = `1.1`.
: Multiplier for covalent radii used to identify molecular bonds. Based on
  [`ase.neighborlist.natural_cutoffs`](https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.natural_cutoffs).
  Usually no change needed.

`stoichiometry` : `list[int]` | default = `[1]`.
: Molecular stoichiometry. `[1]` for a single-component crystal.

  ```{note}
  Multi-component crystals are coming soon!
  ```

`seed` : `int` | default = `42`.
: Random seed for reproducibility.

`lattice_norm_dev` : `float` | default = `0.4`.
: Standard deviation for lattice norm sampling.

`lattice_angle_std` : `float` | default = `8`.
: Standard deviation (degrees) for lattice angle sampling.
