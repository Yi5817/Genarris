"""
This module implements the molecule bonding module.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np
import networkx as nx
from ase.io import read
from ase.data import atomic_numbers, vdw
from ase.neighborlist import NeighborList, natural_cutoffs
# Update Hydrogen radii - set it to 1.1A instead of 1.2A
# Reasoning is here: https://pubs.acs.org/doi/full/10.1021/jp953141%2B
vdw_radii = vdw.vdw_radii.copy()
vdw_radii[1] = 1.1
import gnrs.output as gout

intermolecular_dist = {
    "vdw": {},
    # These are cutoff distances for specific 3 body hydrogen bonds
    "h_bond": {"OH-O": 1.4, "OH-N": 1.5, "NH-O": 1.5, "NH-N": 1.65},
}

def construct_pair_keys(elements: list[str]) -> np.ndarray:
    """
    Constructs pair_key matrix for every element in the argument
    """
    atom_keys = np.array(elements)
    pair_keys = np.char.add(atom_keys, np.array(["-"], dtype="<U2"))
    pair_keys = np.char.add(pair_keys, atom_keys[:, None])
    return pair_keys

def construct_intermolecular_dist_dict() -> None:
    """
    Constructs the intermolecular distance dictionary with vdW radii.
    """
    global intermolecular_dist

    # Construct vdw contact distances for pairwise_dist dictionary
    pair_keys = construct_pair_keys([x for x in atomic_numbers.keys()])
    pair_keys = pair_keys.ravel()

    for pair in pair_keys:
        a1 = atomic_numbers[pair.split("-")[0]]
        a2 = atomic_numbers[pair.split("-")[1]]
        try:
            r1 = vdw_radii[a1]
            r2 = vdw_radii[a2]
        except IndexError:
            continue
        if np.isnan(r1) or np.isnan(r2):
            continue
        intermolecular_dist["vdw"][pair] = r1 + r2
        
construct_intermolecular_dist_dict()

def get_vdw_distance_cutoff_matrix(mol_path: str | list[str], z: int, sr: float, natural_cutoff_mult: float) -> tuple[np.ndarray, list[str]]:
    """
    Get the van der Waals distance cutoff matrix.

    Args:
        mol_path: Path to the molecule file.
        z: Number of molecules in the unit cell.
        sr: Fraction of the van der Waals distance cutoff.
        natural_cutoff_mult: Multiplier for the natural cutoffs.

    Returns:
        cutoff_matrix: Van der Waals distance cutoff matrix.
        hbond_keys: List of hydrogen bond keys.
    """
    if isinstance(mol_path, list):
        mol = []
        for pth in mol_path:
            m = read(pth)
            mol.append(m)
    else:
        mol = read(mol_path)

    mb = MoleculeBonding(mol, natural_cutoff_mult=natural_cutoff_mult)
    cutoff_matrix = mb.get_crystal_cutoff_matrix(z, vdw_mult=sr)
    cutoff_matrix = np.array(cutoff_matrix, dtype="float32")
    hbond_keys = np.unique(mb.hbond_key).tolist()
    return cutoff_matrix, hbond_keys


class MoleculeBonding:
    """
    Identifies all bonded atoms in a molecule or
        structure using ASE NeighborList

    Arguments
    ---------
    atoms: ASE atoms object OR list of ASE atoms object
        Structure object to build molecule bonding
    natural_cutoff_mult: float
        Multiplier to covalent distances for identifying molecular bonding
    skin: float
        For ase.neighborlists. If skin is not zero, then extra neighbors
        outside the cutoff can be returned.

    """

    def __init__(self, atoms, natural_cutoff_mult=1.2, skin=0):
        self.atoms = atoms
        self.hbond_key = np.array([])
        if isinstance(self.atoms, list):
            ele_list = []
            self.bonding_list = np.array([])
            mol_idx = 0
            for mol in self.atoms:
                mol_bond = self._get_bonding(
                    mol, natural_cutoff_mult=natural_cutoff_mult
                )
                self.bonding_list = np.concatenate(
                    (self.bonding_list, mol_bond + mol_idx)
                )
                ele_list += mol.get_chemical_symbols()
                mol_idx += len(mol)
            self.ele = np.array(ele_list, dtype=str)

        else:
            self.bonding_list = self._get_bonding(
                atoms, natural_cutoff_mult=natural_cutoff_mult
            )
            self.ele = np.array(atoms.get_chemical_symbols())

        # Donor and acceptor elements for identifying hydrogen bonds
        self.donor_elements = ["N", "O"]
        self.acceptor_elements = ["N", "O"]

    def get_cutoff_matrix(self, vdw_mult=0.85):
        """
        Returns NxN matrix expressing a cutoff distance between intermolecular
        contacts in a crystal system of the given molecule.

        Arguments
        ---------
        vdw_mult: float
            Multiplicative factor for the cutoff distance
                for vdw type contacts.
            A cutoff value of 0.85 is well supported by statistical analysis
            for these types of contacts.
        """
        # Get pair_key_matrix and construct initial vdw cutoff
        pair_key_matrix = construct_pair_keys(self.ele)
        cutoff_matrix = np.zeros(pair_key_matrix.shape)
        for index, value in np.ndenumerate(pair_key_matrix):
            cutoff_matrix[index] = intermolecular_dist["vdw"][value]
        cutoff_matrix = cutoff_matrix * vdw_mult

        # Add hydrogen bond cutoff distances
        donor_idx, acceptor_idx = self._get_hydrogen_bond_idx()
        gout.emit(f"Donor atom indices =  {donor_idx}")
        gout.emit(f"Acceptor atom indices = {acceptor_idx}")
        if len(donor_idx) > 0:
            acceptor_ele = self.ele[acceptor_idx]
            donor_ele = self.ele[np.concatenate(self.bonding_list[donor_idx])]
            hbond_key = np.char.add(donor_ele, "H-")
            self.hbond_key = np.char.add(hbond_key[:, None], acceptor_ele)

            hbond_values = np.zeros(self.hbond_key.shape)
            for index, value in np.ndenumerate(self.hbond_key):
                hbond_values[index] = intermolecular_dist["h_bond"][value]

            # Create pairwise index grid for indexing into cutoff_matrix
            ixgrid1 = np.ix_(donor_idx, acceptor_idx)
            ixgrid2 = np.ix_(acceptor_idx, donor_idx)
            cutoff_matrix[ixgrid1] = hbond_values
            cutoff_matrix[ixgrid2] = hbond_values.T

        return cutoff_matrix

    def get_cutoff_matrix_vdw(self, vdw_mult=0.85):
        """
        Returns NxN matrix expressing a cutoff distance between intermolecular
        contacts in a crystal system of the given molecule. Only uses vdw
        contact distances.

        Arguments
        ---------
        vdw_mult: float
            Multiplicative factor for the cutoff distance
                for vdw type contacts.
            A cutoff value of 0.85 is well supported by statistical analysis
            for these types of contacts.
        """
        # Get pair_key_matrix and construct initial vdw cutoff
        pair_key_matrix = construct_pair_keys(self.ele)
        cutoff_matrix = np.zeros(pair_key_matrix.shape)
        for index, value in np.ndenumerate(pair_key_matrix):
            cutoff_matrix[index] = intermolecular_dist["vdw"][value]
        cutoff_matrix = cutoff_matrix * vdw_mult
        return cutoff_matrix

    def get_crystal_cutoff_matrix(self, nmpc, vdw_mult=0.85):
        """
        Copies the intermolecular distance matrix from
        MoleculeBonding.get_cutoff_matrix into the correct size for a specific
        number of molecules in the unit cell.
        """
        cutoff_matrix = self.get_cutoff_matrix(vdw_mult=vdw_mult)
        return np.tile(cutoff_matrix, (nmpc, nmpc))

    def _get_bonding(self, atoms, natural_cutoff_mult=1, skin=0):
        # Use ASE neighborlist class to identify bonding of atoms

        cutOff = natural_cutoffs(atoms, mult=natural_cutoff_mult)

        neighborList = NeighborList(
            cutOff, self_interaction=False, bothways=True, skin=0
        )
        neighborList.update(atoms)

        # Construct bonding list indexed by atom in struct
        bonding_list = [[] for x in range(len(atoms))]
        for i in range(len(atoms)):
            bonding_list[i] = neighborList.get_neighbors(i)[0]

        bonding_list = np.array(bonding_list, dtype=object)
        return bonding_list

    def _get_hydrogen_bond_idx(self):
        """
        For all hydrogen atoms in the system, identify the index of any which
        can participate in hydrogen. This is defined as hydrogens which are
        bonded to polar elements: nitrogen, oxygen. Returns a list of
        indices for donors and acceptors of hydrogen bonds
        """
        self.donor_idx = []
        self.acceptor_idx = []

        for i, ele in enumerate(self.ele):
            # Donors definition
            if ele == "H":
                bond_idx = self.bonding_list[i]
                elements = self.ele[bond_idx]
                for h_ele in self.donor_elements:
                    if h_ele in elements:
                        self.donor_idx.append(i)
                        break

            # Acceptor definition
            elif ele in self.acceptor_elements:
                bonding = self.bonding_list[i]
                # Check for terminal oxygen or bridging oxygen
                if ele == "O":
                    if len(bonding) == 1:
                        self.acceptor_idx.append(i)
                    elif len(bonding) == 2:
                        bond_ele = self.ele[bonding]
                        unique_ele = np.unique(bond_ele)
                        if len(unique_ele) == 1 and unique_ele[0] == "C":
                            self.acceptor_idx.append(i)
                        elif len(unique_ele) == 1 and unique_ele[0] == "H":
                            self.acceptor_idx.append(i)

                # Check for terminal nitrogen
                if ele == "N":
                    if len(bonding) <= 2:
                        self.acceptor_idx.append(i)

        return self.donor_idx, self.acceptor_idx
    
class BondNeighborhood:
    """
    Returns the bonding neighborhood of each atom for a structure. User is
    allowed to define a radius that the algorithm traverses to build
    the neighborhood for each atom. If the radius is 0, this would
    correspond to just return the atoms im the system.

    Arguments
    ---------
    radius: int
        Radius is the number of edges the model is allowed to traverse on the
        graph representation of the molecule.
    mb_kwargs: dict
        Keyword arguments for the molecule bonding module. The default values
        are the recommended settings. A user may potentially want to decrease
        the natural_cutoff_mult. This value is multiplied by covalent bond
        radii in the MoleculeBonding class. It's highly recommended that the
        skin value is kept at 0.
        For more details see ibslib.molecules.MoleculeBonding

    """

    def __init__(self, radius=1, mb_kwargs={"natural_cutoff_mult": 1.2, "skin": 0}):
        self.radius = radius
        self.mb_kwargs = mb_kwargs
        if radius != 1:
            raise Exception("Radius greater than 1 not implemented")

    def calc(self, struct):
        self.struct = struct
        self.ele = struct.geometry["element"]
        g = self._build_graph(struct)
        n = self._calc_neighbors(g)
        n = self._sort(g, n)
        fragments = self._construct_fragment_list(n)
        fragments, count = np.unique(fragments, return_counts=True)

        return fragments.tolist(), count

    def _build_graph(self, struct):
        """
        Builds networkx graph of a structures bonding.
        """

        mb = MoleculeBonding(struct.get_ase_atoms(), **self.mb_kwargs)
        g = nx.Graph()

        # Add node to graph for each atom in struct
        self.ele = struct.geometry["element"]
        g.add_nodes_from(range(len(self.ele)))

        # Add edges
        for i, bond_list in enumerate(mb.bonding_list):
            [g.add_edge(i, x) for x in bond_list]

        return g

    def _calc_neighbors(self, g):
        """
        Calculates neighbors for each node in the graph. Uses the radius
        which was declared when BondNeighborhood was initialized. Ordering
        of the neighbors follows:
            1. Terminal atom alphabetical
            2. Self
            3. Bonded atom alphabetical
            4. If continuing from bonded group, print terminal groups of the
               bonded group.
            5. If there's ambiguity about which should come next, place in
               alphabetical order. When the radius is small, there's less
               ambiguity. If the radius becomes large there will be more.
               Although, only a radius of 1 is currently implemented.

        """
        neighbors = [[[]] for x in g.nodes]

        for i, idx_list in enumerate(neighbors):
            neighbor_list = [x for x in g.adj[i]]
            neighbor_list.append(i)
            idx_list[0] += neighbor_list

        return neighbors

    def _sort(self, g, neighbor_list):
        """
        Sorts neighborlist according to definition in _calc_neighbors. Only
        works for a radius of 1.

        Arguments
        ---------
        g: nx.Graph
        neighbor_list: list of int
            List of adjacent nodes plus the node itself as i

        """
        sorted_list_final = [[[]] for x in g.nodes]
        for i, temp in enumerate(neighbor_list):
            # Preparing things which aren't writting well
            idx_list = temp[0]
            current_node = i

            terminal_groups = []
            bonded_groups = []
            for idx in idx_list:
                if g.degree(idx) == 1:
                    terminal_groups.append(idx)
                else:
                    bonded_groups.append(idx)

            terminal_ele = self.ele[terminal_groups]
            alphabet_idx = np.argsort(terminal_ele)
            terminal_groups = [terminal_groups[x] for x in alphabet_idx]

            sorted_list = terminal_groups
            if current_node not in terminal_groups:
                sorted_list.append(current_node)
                remove_idx = bonded_groups.index(current_node)
                del bonded_groups[remove_idx]

            bonded_ele = self.ele[bonded_groups]
            alphabet_idx = np.argsort(bonded_ele)
            bonded_groups = [bonded_groups[x] for x in alphabet_idx]

            sorted_list += bonded_groups

            sorted_list_final[i][0] = sorted_list

        return sorted_list_final

    def _construct_fragment_list(self, n):
        fragment_list = [self.struct.geometry["element"][tuple(x)] for x in n]
        fragment_list = ["".join(x) for x in fragment_list]
        return fragment_list
