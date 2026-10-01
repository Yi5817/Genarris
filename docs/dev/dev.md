# Developer Guide

Contributions are welcome! Please feel free to submit a Pull Request.

## Setting Up for Development

```bash
git clone https://github.com/Yi5817/Genarris.git
cd Genarris
git submodule update --init --recursive
pip install -e .[test]
```

## Running the Tests

```bash
pytest                    # installation and unit tests, a few seconds
pytest -m integration     # full workflows under mpirun, a few minutes
```

The tests live in `tests/`, grouped by what they check:

- `install/` – the package, its compiled `cgenarris` extension, packaged data
  files, task registry and the `gnrs` command are usable with only the core
  dependencies installed.
- `restart/` – checkpoint logs and the restart manager, in a single process.
- `workflow/` – a tiny benzene workflow under `mpirun` exercising restart,
  checkpoint recovery and `--overwrite` end to end. Needs `mpirun` on the
  `PATH` and is skipped otherwise.

Run one group with `pytest tests/install`.

Both sets run in CI on every push and pull request.

## Adding a New Energy Calculator

1. Create a new module under `gnrs/energy/`.
2. Implement a class that inherits from {class}`~gnrs.core.energy.EnergyCalculatorABC`.
3. Register the new calculator in `gnrs/core/registry.py`.

See the {doc}`../api/gnrs.energy` page for the base class interface.

## Adding a New Optimizer

1. Create a new module under `gnrs/optimize/`.
2. Implement a class that inherits from {class}`~gnrs.core.optimizer.GeometryOptimizerABC`.
3. Register the new optimizer in `gnrs/core/registry.py`.
