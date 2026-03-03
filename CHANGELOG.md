# Changelog

## v3.1.0 (2026-03)

All changes in this release by [@Yi5817](https://github.com/Yi5817).

### Highlights

Genarris v3.1.0 brings GPU-accelerated batch optimization, a full Sphinx
documentation site, serial DFT execution, YAML configuration migration,
and bug fixes.

### Added

- **AIMNet2 energy** – AIMNet2 MLIP support for molecular crystal
  energies (`9082ef6`, `c92063d`).
- **Task registry** – centralised `TaskSpec`-based task registration with
  instance IDs, replacing ad-hoc task dispatch (`74f5247`, `e884843`).
- **GPU worker/feeder pattern** – asynchronous GPU worker pool for parallel MLIP
  energy and force evaluations (`f7a65ad`).
- **Batch optimization with GPU mode** – GPU worker/feeder support during
  geometry optimization (`d6b77cb`).
- **Serial DFT mode** – run DFT calculations serially when MPI parallelism is
  not available or desired (`9906837`).
- **Deduplication task** – duplicate crystal removal task integrated into the
  workflow pipeline (`7effd58`, `9958b69`).
- **AP clustering worker limit** – `max_ap_workers` parameter to cap Affinity
  Propagation memory usage (`eb9cf84`).
- **Feature descriptor cleanup** – automatic removal of intermediate feature
  descriptor files after clustering (`0883c8c`).
- **Random seed support** – reproducible runs via a configurable random seed
  (`5130987`).
- **Sphinx documentation site** – full documentation with MyST Markdown,
  Furo theme, configuration reference, API docs, quickstart guide, case studies,
  and GitHub Pages deployment (`dc0ff5e`, `a25b9fa`).
- **Additional geometry optimization tasks** registered in the task system
  (`1274390`).

### Changed

- **JSON → YAML configuration** – migrated all configuration from JSON to YAML
  with updated package data inclusion (`dcc0c67`).
- **Logger refactor** – updated logger throughout the codebase (`9c4335d`).
- **StructureMatcher settings** – updated tolerance settings for more reliable
  structure matching (`fdd3756`).
- **Python version requirement** – bumped minimum Python to 3.10 (`eb35644`).
- **MC volume estimator** – performance-optimized Monte Carlo volume estimation
  (`a684c0d`).
- **Symmetry rigid press tolerance** – tolerance value for symmetry-aware
rigid body pressure optimization (`c11a002`).
- **Project metadata** – updated `pyproject.toml` dependencies, build
  requirements, and license format (`a9e24f1`, `9f32eec`).
- **License** – updated LICENSE file (`301a8ea`, `34150e6`).

### Fixed

- **SWIG ≥ 4.4 compatibility** – avoid compilation errors from wrapper files
  generated with newer SWIG versions (`81b4392`, `fd6bf13`).
- **mpi4py headers** – include mpi4py headers in SWIG options for correct
  compilation (`d7333c0`).
- **Hydrogen bond acceptor** – recognise R-O-H oxygen as a hydrogen bond
  acceptor (`f7b2066`).
- **Unit cell volume std key** – correct configuration key for unit cell volume
  standard deviation (`409740a`).
- **AP clustering default** – remove incorrect default value for `clusters_tol`
  in `APCluster` (`9d39b60`).
- **Structure writing** – improved logic for writing optimised structures to
  disk (`6ff3a7c`).
- **Moment of inertia** – correct axis handling in moment of inertia
  calculations (`8c2c82d`).
- **Experimental structure check** – validate experimental reference structure
  in config before use (`97dd275`).
- **Docstring indentation** – fix indentation issues in module docstrings
  (`ec8b174`).
- **cgenarris docs inclusion** – remove docs inclusion for cgenarris that caused
  build errors (`43d61d9`).

### Documentation

- Comprehensive configuration reference (master, workflow, energy, generation,
  clustering, selection, descriptor, optimisers, rigid press, deduplication,
  experimental) (`a25b9fa`).
- DFT execution modes documentation (serial vs MPI-parallel) (`7491e6d`).
- Installation instructions including pybind11 and mpi4py
  (`92ad061`, `2e2647e`, `61ff130`).
- Quickstart guide (`498b1cc`).
- Citation and references pages (`b37fb57`, `8240237`, `b38892f`).
- Updated README with badges, features, and usage instructions
  (`146f18c`, `30fbcb7`, `a0524b2`, `fac45d0`).

### CI / Chores

- GitHub Pages deployment workflow for documentation (`98e1bf2`, `3f87de3`).
- Updated cgenarris submodule (`d160851`, `6f9fd47`).

---

## v3.0.0 (2025-07)

All changes in this release by [@Yi5817](https://github.com/Yi5817).

### Highlights

Initial public release of Genarris v3.

### Added

- **CLI & workflow orchestration** – `gnrs` CLI entry point with YAML-driven
  workflow engine for end-to-end crystal structure prediction
  (`35a9e2a`, `4960fb8`).
- **UMA energy** – Universal Materials Accelerator (UMA) MLIP support
  (`cf23bb6`, `43cab58`).
- `pandas<=2.3` as optional dependency (`cddeb16`).
- Black linting GitHub Action (`89599c5`).

### Changed

- Simplified MPI compiler detection and build configuration (`350a4e3`).
- Updated `pyproject.toml` dependencies and build setup (`a279dd4`, `10663ec`).
- Removed `requirements.txt` in favour of `pyproject.toml` (`7a37709`).

### Fixed

- Unified logger name to `genarris` (`db6f382`).
- Safe `get` with `False` default for `debug_mode` (`0b17a0a`).
