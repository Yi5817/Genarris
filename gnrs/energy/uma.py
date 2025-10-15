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

import torch
from ase import Atoms
from fairchem.core import pretrained_mlip, FAIRChemCalculator

from gnrs.core.energy import EnergyCalculatorABC


class UMAEnergy(EnergyCalculatorABC):
    """
    Computes the energy using UMA model.
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
        model_name = self.tsk_set.get("model_name", "uma-s-1p1")
        task_name = self.tsk_set.get("task_name", "omc")
        # make sure you applied for model access to the
        # UMA model repository on HuggingFace, and have
        # logged in to Hugging Face using an access token
        self.calc = FAIRChemCalculator(
            pretrained_mlip.get_predict_unit(
                model_name,
                device=device
            ),
            task_name=task_name
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
