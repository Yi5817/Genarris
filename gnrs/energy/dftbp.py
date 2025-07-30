"""
This module computes the energy using DFTB+ code with ASE.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from ase import Atoms
from ase.calculators.dftb import Dftb

from gnrs.core.energy import EnergyCalculatorABC


class DFTBPEnergy(EnergyCalculatorABC):
    """
    Computes energy using DFTB+ code.
    https://wiki.fysik.dtu.dk/ase/ase/calculators/dftb.html
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.tsk_set["energy_settings"]["slako_dir"] = self.tsk_set["sk_files"]
        self.tsk_set["energy_settings"]["command"] = self.tsk_set["command"]
        self.calc = Dftb(**self.tsk_set["energy_settings"])

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
