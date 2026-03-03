"""
This module computes the energy using MACE-OFF model.

https://github.com/ACEsuit/mace
https://github.com/ACEsuit/mace-off

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from ase import Atoms

from gnrs.core.energy import EnergyCalculatorABC


class MACEOFFEnergy(EnergyCalculatorABC):
    """Computes the energy using MACE-OFF model.

    GPU device assignment is managed by the base class via ``GPUDeviceManager``.
    Feeder ranks skip model loading entirely; only GPU workers instantiate
    the MACE calculator.
    """

    requires_gpu = True

    def __init__(self, *args) -> None:
        super().__init__(*args)
        if self._gpu_mgr is None or self._gpu_mgr.is_worker:
            from mace.calculators import mace_off

            model_size = self.tsk_set.get("model_size", "large")
            self.calc = mace_off(model=model_size, device=self.device)

    def initialize(self) -> None:
        """
        Initialize the energy calculator.
        """
        pass

    def compute(self, xtal: Atoms) -> None:
        """
        Compute the energy of the crystal.
        """
        xtal.calc = self.calc
        try:
            energy = xtal.get_potential_energy()
        except Exception:
            energy = 0
        xtal.info[self.energy_name] = energy

    def finalize(self) -> None:
        """
        Finalize the energy calculator.
        """
        pass
