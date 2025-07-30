"""
This module provides the ACSF descriptor implementation.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

from mpi4py import MPI

from ase import Atoms
from ase.io import read
from dscribe.descriptors import ACSF

from gnrs.core.descriptor import DescriptorABC

class ACSFDescriptor(DescriptorABC):
    """
    Computes Atom-Centered Symmetry Function (ACSF) descriptors.
    
    ACSFs can be used to represent the local environment near an atom 
    by using a fingerprint composed of the output of multiple two- and 
    three-body functions that can be customized to detect specific structural features.
    """

    def __init__(self, comm: MPI.Comm, task_settings: dict) -> None:
        """
        Initialize the ACSF descriptor calculator.
        
        Args:
            comm: MPI communicator for parallel computation
            task_settings: Task settings Dictionary for ACSF descriptor
        """
        super().__init__(comm, task_settings)
        mol_path = self.tsk_set.get("molecule_path", None)
        r_cut = self.tsk_set.get("r_cut", None)
        g2_params = self.tsk_set.get("g2_params", None)
        g3_params = self.tsk_set.get("g3_params", None)
        g4_params = self.tsk_set.get("g4_params", None)
        self.vector_pooling = self.tsk_set.get("vector_pooling", None)

        species_list = []
        unique_species = set()
        mol_len = 0

        for mpth in mol_path:
            mol = read(mpth, parallel=False)
            symbols = list(mol.symbols)
            species_list.extend(symbols)
            unique_species.update(symbols)
            mol_len += len(symbols)
        
        self.mol_len = mol_len
        
        # Initialize ACSF descriptor
        self.acsf = ACSF(
            r_cut=r_cut,
            g2_params=g2_params,
            g3_params=g3_params,
            g4_params=g4_params,
            species=unique_species, 
            periodic=True
        )

    def initialize(self) -> None:
        """
        Initialization step.
        """
        pass

    def compute(self, xtal: Atoms) -> None:
        """
        Compute ACSF descriptors for the given crystal structure.
        
        Args:
            xtal: Crystal structure to compute descriptors.
        """
        
        acsf_xtal = self.acsf.create(
            xtal, 
            centers=None, 
            n_jobs=1, 
            verbose=False
        )
        if self.vector_pooling is None:
            xtal.info["acsf"] = acsf_xtal[:self.mol_len].reshape(1, -1)
        elif self.vector_pooling == "mean":
            xtal.info["acsf"] = acsf_xtal.mean(axis=0, keepdims=True)
        elif self.vector_pooling == "sum":
            xtal.info["acsf"] = acsf_xtal.sum(axis=0, keepdims=True)
        elif self.vector_pooling == "max":
            xtal.info["acsf"] = acsf_xtal.max(axis=0, keepdims=True)
        
        del acsf_xtal
        return

    def finalize(self) -> None:
        """
        Finalization step.
        """
        pass
