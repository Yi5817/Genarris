"""
This module computes the energy using FHI-aims DFT code with ASE.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from ase import Atoms
from ase.calculators.aims import Aims, AimsProfile

from gnrs.core.energy import EnergyCalculatorABC
from gnrs.gnrsutil.mpi_cmd import build_dft_command


class AIMSEnergy(EnergyCalculatorABC):
    """
    Computes the energy using FHI-aims DFT code with ASE
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)
        command = build_dft_command(self.tsk_set, self.rank)
        prof = AimsProfile(
            command,
            default_species_directory=self.tsk_set["species_dir"],
        )
        self.calc = Aims(profile=prof, **self.tsk_set['energy_settings'])

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
