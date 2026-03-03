# Selection

Selection runs after clustering. Configure it under the selection section name.

---

## `center`

Selects the cluster exemplar from each cluster.

```ini
[center]
cluster_name    = ap
filter          = center
```

`cluster_name` : `str`.
: Key in `xtal.info` containing the cluster label (e.g. `ap`, `kmeans`).

`filter` : `str` | default = `center`.
: `center` picks the exemplar identified during clustering. Other values are treated as a key in `xtal.info` -- the structure with the minimum value in each cluster is selected.

---

## `window`

Selects structures within an energy window from each cluster.

```ini
[window]
cluster_name          = ap
filter                = lbfgs_maceoff
energy_window         = 10.0
n_structs_per_cluster = None
z                     = 4
```

`cluster_name` : `str`.
: Key in `xtal.info` containing the cluster label.

`filter` : `str`.
: Key in `xtal.info` holding the energy value to rank by.

`energy_window` : `float`.
: Window width in kJ/mol above the cluster minimum.

`n_structs_per_cluster` : `int` | default = `None`.
: Maximum structures to keep per cluster. `None` means no limit.

`z` : `int`.
: Number of molecules per unit cell.
