# Developer Guide

Contributions are welcome! Please feel free to submit a Pull Request.

## Setting Up for Development

```bash
git clone https://github.com/Yi5817/Genarris.git
cd Genarris
git submodule update --init --recursive
pip install -e .
```

## Adding a New Energy Calculator

1. Create a new module under `gnrs/energy/`.
2. Implement a class that inherits from {class}`~gnrs.core.energy.EnergyCalculatorABC`.
3. Register the new calculator in `gnrs/core/registry.py`.

See the {doc}`../api/gnrs.energy` page for the base class interface.

## Adding a New Optimizer

1. Create a new module under `gnrs/optimize/`.
2. Implement a class that inherits from {class}`~gnrs.core.optimizer.GeometryOptimizerABC`.
3. Register the new optimizer in `gnrs/core/registry.py`.
