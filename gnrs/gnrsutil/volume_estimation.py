"""
This module is based on the PyMoVE algorithm implemented in: 
[PyMoVE](https://github.com/manny405/PyMoVE)
predicting the unit cell volume of a molecule.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import yaml
from importlib.resources import files

import numpy as np
from scipy.spatial.distance import cdist
from ase.data import vdw_radii, atomic_numbers
from gnrs.gnrsutil.molecule_bonding import BondNeighborhood
from gnrs.core.structure import Structure

_PYMOVE_MODEL = yaml.safe_load(
    files("gnrs.gnrsutil").joinpath("pymove_model.yaml").read_text()
)
_FRAGMENT_COEF: dict[str, float] = _PYMOVE_MODEL["fragment_coef"]


def predict_cell_volume(mol_path, seed=42) -> float:
    """
    Predict the unit cell volume of a molecule.

    Args:
        mol_path: Path to the molecule file.

    Returns:
        Predicted unit cell volume.
    """
    struct = Structure()
    struct.build_geo_from_atom_file(mol_path)
    mve = MoleculeVolumeEstimator(seed=seed)
    cell_volume = mve.calc(struct)
    return cell_volume


class MoleculeVolumeEstimator:
    """
    Class for performing molecular volume estimation using vdW radii.

    Args:
        tol: Tolerance to converge the estimation.
        iterations: Maximum number of iterations to converge.
        batch: Number of samples to process in MC method at a single time.
        vdW: Array of vdW radii indexed by the atomic number.
        seed: Random seed.
    """

    def __init__(
        self,
        tol: float = 1e-3,
        iterations: int = int(1e8),
        batch: int = 200000,
        vdW: np.ndarray = vdw_radii,
        seed: int = 42,
    ):
        self.iterations = int(iterations)
        self.batch = int(batch)
        self.vdW = vdW
        self.tol = tol
        self.rng = np.random.default_rng(seed)

    def calc(self, molecule_struct: Structure) -> float:
        """
        Calculates the predicted molecular volume for a Structure object.
        """
        volume = self.MC(molecule_struct)
        if volume is None:
            return None
        volume = self.bond_neighberhood_model(molecule_struct, volume)
        return volume

    def bond_neighberhood_model(
        self, molecule_struct: Structure, MC_volume: float
    ) -> float:
        """
        Applies linear correction to the input volume from CSD analysis.
        """
        fragment_coef = _FRAGMENT_COEF
        intercept = 0.0

        # Start with MC_volume contribution + intercept
        volume = MC_volume * fragment_coef["MC_volume"] + intercept

        # Calc structure fragments and counts
        bn = BondNeighborhood()
        fragments, counts = bn.calc(molecule_struct)

        # Accumulate contributions
        for frag, count in zip(fragments, counts):
            if frag in fragment_coef:
                volume += count * fragment_coef[frag]

        return volume

    def MC(self, molecule_struct: Structure) -> float:
        """
        Monte Carlo method for volume estimation using vdW radii.
        """
        geo = molecule_struct.get_geo_array()

        # Get radii array for each element
        radii = np.array(
            [self.vdW[atomic_numbers[ele]] for ele in molecule_struct.geometry["element"]]
        )
        radii_sq = radii ** 2

        # Set sample region
        min_region = np.min(geo, axis=0) - 5.0
        max_region = np.max(geo, axis=0) + 5.0
        region_range = max_region - min_region
        region_volume = float(np.prod(region_range))

        # MC tracking
        vdW_in_total = 0
        total_samples = 0
        prev_volume = None

        for start in range(0, self.iterations, self.batch):
            batch_size = min(self.batch, self.iterations - start)

            # Generate random points, uniform in bounding box
            points = self.rng.uniform(size=(batch_size, 3)) * region_range + min_region
            dist_sq = cdist(geo, points, metric="sqeuclidean")

            # Check if any atom is close enough: dist_sq <= radii^2
            # Broadcasting radii_sq
            inside = np.any(dist_sq <= radii_sq[:, None], axis=0)
            vdW_in_total += np.count_nonzero(inside)
            total_samples += batch_size

            # Check convergence
            current_volume = (vdW_in_total / total_samples) * region_volume
            if prev_volume is not None:
                err = abs(current_volume - prev_volume)
                if err <= self.tol:
                    break
            prev_volume = current_volume

        molecule_volume = (vdW_in_total / total_samples) * region_volume
        return molecule_volume