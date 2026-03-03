# Config Reference

Genarris loads settings from an INI (`.conf` / `.ini`), JSON, or YAML file.
Each section maps to a task or module in the workflow. Options not specified
by the user fall back to the defaults in `gnrs/parser/defaults.yaml`.

## Full Example

A complete end-to-end pipeline: generate structures, optimize with
symmetry-preserving rigid press, deduplicate, evaluate energies with
MACE-OFF, compute ACSF descriptors, cluster via Affinity Propagation and
select representatives, relax with BFGS + MACE-OFF, and deduplicate again.

:::{dropdown} Show full `ui.conf`
:icon: file-code
:open:

```ini
[master]
name                        = aspirin
molecule_path               = ["aspirin.xyz"]
Z                           = 4

[workflow]
tasks                       = ['generation',
                               'symm_rigid_press',
                               'dedup',
                               'maceoff',
                               'acsf',
                               'ap_center',
                               'bfgs_maceoff',
                               'dedup']

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

[dedup]
stol                        = 0.5
ltol                        = 0.5
angle_tol                   = 10

[dedup_2]
stol                        = 0.3
ltol                        = 0.2
angle_tol                   = 10
energy_key                  = maceoff
group_by_spg                = False

[maceoff]
model_size                  = large

[acsf]
r_cut                       = 8.0
g2_params                   = [[1, 1], [1, 2], [1, 3]]
g4_params                   = [[1, 1, 1], [1, 2, 1], [1, 1, -1]]
pca                         = True
n_components                = 0.98

[ap]
feature_name                = acsf_pca
n_clusters                  = 0.1
clusters_tol                = 0.05

[center]
cluster_name                = ap
filter                      = center

[bfgs]
energy_method               = maceoff
fmax                        = 0.01
steps                       = 1000
cell_opt                    = True
fix_sym                     = True
```
:::

## Sections

```{toctree}
:maxdepth: 1

master
workflow
generation
rigidpress
dedup
energy
optimizers
descriptor
clustering
selection
experimental
```
