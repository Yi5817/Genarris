# Workflow

Defines which tasks to run and in what order.

```ini
[workflow]
tasks = ['generation', 'symm_rigid_press', 'dedup', 'maceoff', 'acsf', 'ap_center']
```

`tasks` : `list[str]`.
: Ordered list of task names to execute. See the table below for all
  recognized names.

## Available tasks

```{list-table}
:header-rows: 1
:widths: 28 72

* - Task name
  - Description
* - `generation`
  - Random crystal structure generation
* - `rigid_press`
  - Rigid press geometry optimization (C implementation, fast but may break symmetry)
* - `symm_rigid_press`
  - Symmetry-preserving rigid press geometry optimization
* - `dedup`
  - Duplicate removal via [`pymatgen StructureMatcher`](https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher).
* - `bfgs_<energy_method>`
  - BFGS geometry optimization with an energy calculator (e.g. `bfgs_maceoff`)
* - `lbfgs_<energy_method>`
  - LBFGS geometry optimization with an energy calculator (e.g. `lbfgs_aims`)
* - `energy_method`
  - Single-point energy with `energy_method` Available: `maceoff`, `uma`, `aimnet`, `aims`, `vasp`, `dftb`.
* - `acsf`
  - Atom-centered symmetry function descriptor evaluation.
* - `ap_center`
  - Affinity Propagation clustering + center selection
* - `ap_window`
  - Affinity Propagation clustering + energy-window selection
* - `kmeans_center`
  - K-Means clustering + center selection
* - `kmeans_window`
  - K-Means clustering + energy-window selection
```

Duplicate task names are auto-indexed (e.g. two `dedup` entries become
`dedup_1` and `dedup_2`) so each step gets its own output directory.
