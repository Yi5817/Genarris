# Descriptor

## `acsf`

Atom-Centered Symmetry Functions (ACSF) descriptor from [DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/acsf.html).

```ini
[acsf]
r_cut        = 8.0
g2_params    = [[1, 1], [1, 2], [1, 3]]
g4_params    = [[1, 1, 1], [1, 2, 1], [1, 1, -1]]
pca          = True
n_components = 0.98
```

`r_cut` : `float`.
: The smooth cutoff value in angstroms.

`g2_params` : `list[list]` | default = `None`.
: A list of pairs of $\eta$ and $R_s$ parameters for $G^2$ functions.

`g3_params` : `list[list]` | default = `None`.
: A list of $\kappa$ parameters for $G^3$ functions.

`g4_params` : `list[list]` | default = `None`.
: A list of triplets of $\eta$, $\zeta$ and $\lambda$ parameters for $G^4$ functions.

```{note}
Check out the [`ACSF.__init__`](https://singroup.github.io/dscribe/latest/tutorials/descriptors/acsf.html#dscribe.descriptors.acsf.ACSF.__init__) for more details.
```

`vector_pooling` : `str` | default = `None`.
: Pooling across atoms: `None` (per-atom), `mean`, `sum`, or `max`.

`pca` : `bool` | default = `False`.
: Apply PCA compression to the ACSF vector.

`n_components` : `float` | `int` | default = `None`.
: Number of components to keep. Only used when `pca` is `True`. 

```{note}
Check out the [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for more details.
```
