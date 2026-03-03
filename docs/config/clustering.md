# Clustering

Clustering sections define how structures are grouped. The workflow task name
combines the clustering method and the selection method
(e.g. `ap_center`, `kmeans_window`).

---

## `ap`

[Affinity Propagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html)
clustering.

```ini
[ap]
feature_name            = acsf_pca
n_clusters              = 100
damping                 = 0.5
max_iter                = 200
convergence_iter        = 15
max_sampling_attempts   = 10
preference_range        = quantile
clusters_tol            = 5
max_ap_attempts         = 10
max_ap_workers          = 8
debug_mode              = False
```

`feature_name` : `str`.
: Key in `xtal.info` saving the feature vector (e.g. `acsf_pca`, `acsf`).

`n_clusters` : `int` | `float`.
: Target number of clusters. A float in (0, 1) is a fraction of the pool size. A integer is the absolute number of clusters.

`damping` : `float` | default = `0.5`.
: Damping factor for AP (between 0.5 and 1.0).

`max_iter` : `int` | default = `200`.
: Maximum AP iterations.

`convergence_iter` : `int` | default = `15`.
: Iterations with no change before declaring convergence.

`max_sampling_attempts` : `int` | default = `10`.
: Outer iterations to adjust preference and reach `n_clusters`.

`preference_range` : `str` | `list[float]` | default = `quantile`.
: Initial preference range. available options are `quantile` and `mean-median`.

`clusters_tol` : `int` | `float` | default = `0.5`.
: Tolerance on the final cluster count. An integer is absolute; a float is relative to `n_clusters`.

`max_ap_attempts` : `int` | default = `10`.
: Maximum retries per AP worker when clustering fails to converge.

`max_ap_workers` : `int` | default = `8`.
: Maximum MPI ranks used as AP workers. Additional ranks wait.

`debug_mode` : `bool` | default = `False`.
: Print per-iteration AP diagnostics.

---

## `kmeans`

[MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
clustering.

```ini
[kmeans]
feature_name    = acsf_pca
n_clusters      = 100
save_info       = False
```

`feature_name` : `str`.
: Key in `xtal.info` saving the feature vector (e.g. `acsf_pca`, `acsf`).

`n_clusters` : `int` | `float`.
: Target number of clusters. A float in (0, 1) is a fraction of the pool size. A integer is the absolute number of clusters.

`save_info` : `bool` | default = `False`.
: Store per-structure distance to cluster center in `xtal.info`.
