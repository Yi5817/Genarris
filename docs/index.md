# Genarris

```{image} assets/images/Genarris_logo.png
:alt: Genarris Logo
:width: 360px
:align: center
```

<p align="center" class="badges">
<a href="https://github.com/Yi5817/Genarris/releases"><img src="https://img.shields.io/badge/version-3.1.0-green" alt="Version 3.1.0"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
<a href="https://github.com/Yi5817/Genarris/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue" alt="License"></a>
<a href="https://doi.org/10.1021/acs.jctc.5c01080"><img src="https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.5c01080-blue" alt="DOI"></a>
</p>

*A scalable, MPI-parallel Crystal Structure Prediction workflow for organic molecular crystals.*

🏛️ Developed by the [Noa Marom Group](https://www.noamarom.com/software/genarris) at Carnegie Mellon University.

---

::::{grid} 1 1 2 4
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: user/install
:link-type: doc
:class-header: bg-light

**📦 Installation**
^^^
Set up Genarris with pip in minutes.
:::

:::{grid-item-card}
:link: user/quickstart
:link-type: doc
:class-header: bg-light

**🚀 Quick Start**
^^^
Run your first CSP workflow step by step.
:::

:::{grid-item-card}
:link: api/index
:link-type: doc
:class-header: bg-light

**📖 API Reference**
^^^
Learn about `gnrs` API.
:::

:::{grid-item-card}
:link: about/case_studies
:link-type: doc
:class-header: bg-light

**💎 Case Studies**
^^^
CSP results and applications.
:::
::::

## Key Features

::::{grid} 2 2 3 3
:gutter: 3

:::{grid-item-card} 🔬 Structure Generation
Random crystal generation across all 230 space groups.
:::

:::{grid-item-card} 📐 Rigid Press
Improve efficiency of close-packed molecular crystal generation.
:::

:::{grid-item-card} 🧠 ML Potentials
Evaluate energies with state-of-the-art MLIPs including UMA, MACE-OFF, and AIMNet2 with GPU acceleration.
:::

:::{grid-item-card} 📊 Clustering & Selection
AP / K-Means clustering with ACSF descriptors and flexible selection strategies.
:::

:::{grid-item-card} ⚡ MPI Parallel
Scales to hundreds of cores with GPU worker/feeder pattern.
:::

:::{grid-item-card} 🧩 Modular Workflow
Configurable CSP workflows with extensible base classes.
:::

::::

## Supported Energy Calculators

```{list-table}
:header-rows: 1
:widths: 18 16 10 56
:class: calc-table

* - 🔧 Calculator
  - 🏷️ Type
  - 🖥️ GPU
  - 📝 Description
* - [UMA](https://github.com/facebookresearch/fairchem)
  - {bdg-success}`MLIP`
  - ✅
  - Universal Model for Atoms from [Meta FAIR](https://ai.meta.com/research/) Chemistry Team.
* - [MACE-OFF](https://github.com/ACEsuit/mace)
  - {bdg-success}`MLIP`
  - ✅
  - Transferable Organic Force Fields
* - [AIMNet2](https://github.com/isayevlab/aimnetcentral)
  - {bdg-success}`MLIP`
  - ✅
  - Flexible long-range interactions
* - [DFTB+](https://dftbplus.org/)
  - {bdg-warning}`Semi-Empirical`
  - —
  - Density Functional Tight Binding
* - [FHI-aims](https://fhi-aims.org/)
  - {bdg-primary}`DFT`
  - —
  - All-electron DFT code
* - [VASP](https://www.vasp.at/)
  - {bdg-primary}`DFT`
  - —
  - Plane-wave DFT code
```

## Citation

:::{admonition} Please cite
:class: tip

Yang, Y., Tom, R., Wui, J. A., Moussa, J. E., & Marom, N.
Genarris 3.0: Generating Close-Packed Molecular Crystal Structures with Rigid Press.
*Journal of Chemical Theory and Computation*, **21**, 11318–11332 (2025).

See the full {doc}`about/citation` page for all related papers.
:::

```{toctree}
:caption: Quick Start
:hidden:
user/install
user/quickstart
```

```{toctree}
:caption: Reference
:hidden:
config/index
api/index
```

```{toctree}
:caption: Developer Guide
:hidden:
dev/dev
```

```{toctree}
:caption: About
:hidden:

about/case_studies
about/changelog
about/citation
about/license
```
