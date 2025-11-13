"""
This module computes the energy using AIMNet2 model.

https://github.com/isayevlab/aimnetcentral

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import torch
from ase import Atoms
from aimnet.calculators import AIMNet2ASE

from gnrs.core.energy import EnergyCalculatorABC


class AIMNETEnergy(EnergyCalculatorABC):
    """
    Computes the energy using AIMNet model.
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_id = self.rank % num_gpus
                device = f"cuda:{gpu_id}"
                torch.cuda.set_device(gpu_id)
            else:
                device = "cpu"
        else:
            device = "cpu"
        model = self.tsk_set.get("model", "aimnet2")
        self.calc = AIMNet2ASE(model)
        self.calc.base_calc.set_lrcoulomb_method('ewald')

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
