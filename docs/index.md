# Genarris

```{image} assets/images/Genarris_logo.png
:alt: Genarris Logo
:width: 360px
:align: center
```

<p align="center" class="badges">
<a href="https://github.com/Yi5817/Genarris/releases"><img src="https://img.shields.io/badge/version-3.0.0-green" alt="Version 3.0.0"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"></a>
<a href="https://github.com/Yi5817/Genarris/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue" alt="License"></a>
<a href="https://doi.org/10.1021/acs.jctc.5c01080"><img src="https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.5c01080-blue" alt="DOI"></a>
</p>

*A scalable, MPI-parallel Crystal Structure Prediction workflow for organic molecular crystals.*

**Genarris** (`gnrs`) generates random molecular crystal structures and drives
CSP workflows from generation through energy evaluation, clustering,
and selection. 

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
Full module and class documentation.
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
Generate random molecular crystal structures across all 230 space groups with
configurable packing parameters and volume estimation.
:::

:::{grid-item-card} 📐 Rigid Press
Improve packing acceptance rates with symmetry-preserving rigid-press geometry
optimization.
:::

:::{grid-item-card} 🧠 ML Potentials
Evaluate energies with state-of-the-art MLIPs including **UMA**, **MACE-OFF**,
and **AIMNet2** with GPU acceleration.
:::

:::{grid-item-card} 📊 Clustering & Selection
Cluster structures using Affinity Propagation or K-Means with ACSF descriptors.
Select representatives via center or energy-window strategies.
:::

:::{grid-item-card} ⚡ MPI Parallelization
Scale across hundreds of cores with MPI-parallel execution. GPU worker/feeder
pattern for efficient resource utilization.
:::

:::{grid-item-card} 🧩 Modular Workflow
Configure multi-step CSP pipelines via simple INI files. Extensible architecture
with abstract base classes for custom implementations.
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
  - Universal Model for Atoms from Meta FAIR
* - [MACE-OFF](https://github.com/ACEsuit/mace)
  - {bdg-success}`MLIP`
  - ✅
  - MACE-OFF organic molecular crystals model
* - [AIMNet2](https://github.com/isayevlab/aimnetcentral)
  - {bdg-success}`MLIP`
  - ✅
  - AIMNet2 neural network potential
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
about/examples
about/changelog
about/citation
about/license
```
