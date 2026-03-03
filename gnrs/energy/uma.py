"""
This module computes the energy using the Universal Model for Atoms (UMA) model from fairchem.

https://github.com/facebookresearch/fairchem
https://fair-chem.github.io/
https://huggingface.co/facebook/UMA

Models are made accessible for commerical and non-commerical use under a permissive license 
found in https://huggingface.co/facebook/UMA/blob/main/LICENSE.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Vahe Gharakhanyan"]
__email__ = "vaheg@meta.com"

from ase import Atoms

from gnrs.core.energy import EnergyCalculatorABC


class UMAEnergy(EnergyCalculatorABC):
    """Computes the energy using UMA model.

    GPU device assignment is managed by the base class via ``GPUDeviceManager``.
    Feeder ranks skip model loading entirely; only GPU workers instantiate
    the FAIRChem calculator.
    """

    requires_gpu = True

    def __init__(self, *args) -> None:
        super().__init__(*args)
        if self._gpu_mgr is None or self._gpu_mgr.is_worker:
            from fairchem.core import pretrained_mlip, FAIRChemCalculator

            model_name = self.tsk_set.get("model_name", "uma-s-1p1")
            task_name = self.tsk_set.get("task_name", "omc")
            self.calc = FAIRChemCalculator(
                pretrained_mlip.get_predict_unit(
                    model_name,
                    device=self.device,
                ),
                task_name=task_name,
            )

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
