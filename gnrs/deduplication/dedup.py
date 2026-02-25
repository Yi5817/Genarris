"""
Duplicate structure removal using pymatgen StructureMatcher.

Structures are grouped by space group for computational efficiency,
then within each space group a reference structure is broadcast to all MPI ranks
and compared against the remaining candidates in parallel.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
import random
from collections import defaultdict

from ase.atoms import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

import gnrs.parallel as gp

logger = logging.getLogger("dedup")


def group_by_spg(structs: dict[str, Atoms]) -> dict[int, dict[str, Atoms]]:
    """
    Group structures by space group.

    Args:
        structs: {name: Atoms}.

    Returns:
        {spg: {name: Atoms, ...}}.
    """
    groups: dict[int, dict[str, Atoms]] = defaultdict(dict)
    for name, xtal in structs.items():
        spg = xtal.info.get("spg")
        groups[spg][name] = xtal
    return groups


def _select(
    candidates: dict[str, Atoms],
    energy_key: str | None
) -> str:
    """
    Select one structure from a set of duplicates.

    If energy_key is provided, the lowest-energy structure is chosen. 
    Otherwise a random one is chosen.

    Args:
        candidates: {name: Atoms} duplicates.
        energy_key: Key in Atoms.info for energy, or None.

    Returns:
        Name of the chosen structure.
    """
    if energy_key is not None:
        energies = []
        for name, xtal in candidates.items():
            e = xtal.info.get(energy_key)
            if e is not None:
                energies.append((name, float(e)))
        if len(energies) == len(candidates):
            return min(energies, key=lambda x: x[1])[0]

    return random.choice(sorted(candidates.keys()))

def _scatter_structs(pool: dict[str, Atoms]) -> dict[str, Atoms]:
    """
    Master scatters a dict of structures evenly across ranks.
    """
    scatter_list = None
    if gp.is_master:
        items = list(pool.items())
        n = len(items)
        per_rank = n // gp.size
        remainder = n % gp.size
        scatter_list = []
        start = 0
        for r in range(gp.size):
            chunk = per_rank + (1 if r < remainder else 0)
            scatter_list.append(dict(items[start : start + chunk]))
            start += chunk
    return gp.comm.scatter(scatter_list, root=0)


def dedup_group(
    pool: dict[str, Atoms],
    matcher: StructureMatcher,
    spg: int | None,
    energy_key: str | None,
) -> dict[str, Atoms]:
    """
    Remove duplicates from a space group in parallel.

    1. Master picks one candidate from the pool and broadcasts its
        pymatgen Structure to all ranks.
    2. The remaining structures are scattered across ranks; each rank
        tests matcher.fit(candidate, local_struct) in parallel.
    3. Match results are gathered. Master collects the duplicate
        cluster, selects the best structure, and removes
        duplicates from the pool until the pool is empty.

    Args:
        pool: {name: Atoms} — all structures in this space group
            (only meaningful on master; ignored on workers).
        matcher: Configured StructureMatcher instance.
        spg: Space group.
        energy_key: Key in Atoms.info for energy, or None.

    Returns:
        {name: Atoms} — unique structures in the space group.
    """
    kept = {}

    while True:
        n_rem = len(pool) if gp.is_master else 0
        n_rem = gp.comm.bcast(n_rem, root=0)
        if n_rem == 0:
            break

        if gp.is_master:
            ref_name = next(iter(pool))
            ref_xtal = pool.pop(ref_name)
            pmg_ref = AseAtomsAdaptor.get_structure(ref_xtal)
        else:
            ref_name = None
            ref_xtal = None
            pmg_ref = None

        ref_name = gp.comm.bcast(ref_name, root=0)
        ref_xtal = gp.comm.bcast(ref_xtal, root=0)
        pmg_ref = gp.comm.bcast(pmg_ref, root=0)

        local_chunk = _scatter_structs(pool if gp.is_master else {})

        local_matches = []
        for name, xtal in local_chunk.items():
            pmg_xtal = AseAtomsAdaptor.get_structure(xtal)
            if matcher.fit(pmg_ref, pmg_xtal):
                local_matches.append(name)

        all_matches = gp.comm.gather(local_matches, root=0)

        if gp.is_master:
            match_names = set()
            for sublist in all_matches:
                match_names.update(sublist)

            cluster = {ref_name: ref_xtal}
            for mn in match_names:
                cluster[mn] = pool.pop(mn)

            best = _select(cluster, energy_key)
            kept[best] = cluster[best]

            logger.debug(
                f"SPG {spg}: remaining pool: {len(pool)}"
            )

    kept = gp.comm.bcast(kept, root=0)
    return kept
