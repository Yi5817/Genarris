"""
This module contains the DistributedStructs class, which is used to handle distributed structure dictionaries.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import json
import logging
import os
from itertools import chain
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from ase.io.jsonio import decode, encode

import gnrs.output as gout
import gnrs.parallel as gp

logger = logging.getLogger("DistributedStructs")


class DistributedStructs:
    """
    Contains functions for handling distributed structure dictionaries.
    """

    def __init__(self, structs: dict):
        """
        Initialize with a dictionary of structures.
        
        Args:
            structs: Dictionary mapping structure names to ASE Atoms objects
        
        Raises:
            ValueError: If structs is not a dictionary
        """
        self.structs = structs
        self.logger = logger

        self._checkpointed: set[str] = set()

    def get_num_structs(self) -> int:
        """
        Get the total number of structures in a distributed
        structures dictionary.
        
        Returns:
            Total number of structures across all ranks
        """
        # Handle structs = None case
        if self.structs is not None:
            num_each_rank = len(self.structs)
        else:
            num_each_rank = 0

        total_num_list = gp.comm.gather(num_each_rank, root=0)
        if gp.is_master:
            total_num = sum(total_num_list)
            logger.debug(f"xtal distribution across cores: {total_num_list}")
        else:
            total_num = None

        total_num = gp.comm.bcast(total_num, root=0)
        return total_num

    def find_matches(self, target: Atoms, settings: dict | None = None) -> list:
        """
        Runs pymatgen duplicate checks on a distributed struct dictionary.
        
        Args:
            target: Target structure to be matched
            settings: Settings for pymatgen StructureMatcher
            
        Returns:
            List of matching structure IDs
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.structure_matcher import StructureMatcher

        pmg_target = AseAtomsAdaptor.get_structure(target)

        if settings is None:
            settings = {"stol": 0.5, "ltol": 0.5, "angle_tol": 10}
        matcher = StructureMatcher(**settings)

        match_list = []
        for name, xtal in self.structs.items():
            pmg_xtal = AseAtomsAdaptor.get_structure(xtal)
            if matcher.fit(pmg_target, pmg_xtal):
                match_list.append(name)

        # Combine match list and flatten
        match_list = gp.comm.allgather(match_list)
        match_list = [item for sublist in match_list for item in sublist]

        logger.info(f"Matched with {len(match_list)} structures")
        logger.debug(f"Matched structures = {match_list}")

        return match_list

    def collect_property(self, prpty: str, ptype: str = "info") -> list:
        """
        Collects the property of all the structures into a list.
        
        Args:
            prpty: Property name to collect
            ptype: Property type, either 'info' or 'method'
            
        Returns:
            List of property values on master rank, None on other ranks
        """
        # Construct property list for each rank
        prpty_list = []
        for xtal in self.structs.values():
            if ptype == "method":
                prop = getattr(xtal, prpty)()
            elif ptype == "info":
                prop = xtal.info.get(prpty)
            prpty_list.append(prop)

        # Combine
        prpty_list = gp.comm.gather(prpty_list)
        if gp.is_master:
            prpty_list = [item for sublist in prpty_list for item in sublist]
        return prpty_list

    def get_statistics(self, prpty: str, ptype: str = "info") -> dict:
        """
        Gets the statistics on a property of interest.
        
        Args:
            prpty: Property name to analyze
            ptype: Get property from either 'info' or 'method'
            
        Returns:
            Dictionary with statistics on master rank, None on other ranks
        """
        prpty_list = self.collect_property(prpty, ptype)
        
        if not gp.is_master:
            return None
            
        prpty_array = np.array(prpty_list)
        stats = {
            "Minimum": np.min(prpty_array),
            "Maximum": np.max(prpty_array),
            "Average": np.average(prpty_array),
            "Std": np.std(prpty_array),
        }
        return stats

    def find_spg(self, tol: float = 0.001) -> None:
        """
        Finds the space group of all structures.
        Space group number is stored in info["spg"]
        
        Args:
            tol: Tolerance for symmetry finding
        """
        from ase.spacegroup.spacegroup import get_spacegroup

        for struct in self.structs.values():
            struct.info["spg"] = get_spacegroup(struct, symprec=tol).no

    def checkpoint_save(
        self,
        path: str,
        result_key: str,
        structs: dict[str, Atoms] | None = None,
    ) -> None:
        """
        Appends newly completed structures to this rank's checkpoint log.
        Does not communicate, so ranks can call it independently.

        Every completed structure is written exactly once, as one line in the
        same ``"name": <ase json>,`` form used inside structures.json. A job
        killed mid-write can therefore damage only the last line.

        Args:
            path: Directory for this rank's checkpoint log
            result_key: ``Atoms.info`` key that marks a structure as done
            structs: Structures to scan instead of this rank's own pool, e.g.
                copies of other ranks' structures computed on this rank
        """
        if structs is None:
            structs = self.structs
        done = [
            (name, xtal)
            for name, xtal in structs.items()
            if result_key in xtal.info and name not in self._checkpointed
        ]
        if not done:
            return
        with open(os.path.join(path, f"{gp.rank}.ckpt"), "ab+") as chk:
            # A line cut short by a job kill has no newline; start on a fresh
            # line so that only the damaged entry is lost, not the next one
            chk.seek(0, os.SEEK_END)
            if chk.tell() > 0:
                chk.seek(-1, os.SEEK_END)
                if chk.read(1) != b"\n":
                    chk.write(b"\n")
            for name, xtal in done:
                chk.write(f'"{name}": {encode(xtal)},\n'.encode())
        self._checkpointed.update(name for name, _ in done)

    @staticmethod
    def checkpoint_clear(path: str) -> None:
        """
        Removes checkpoint files left under a task's calc directory.
        Call on one rank only; does not communicate.

        Args:
            path: Directory containing rank_* checkpoint folders
        """
        for pattern in ("rank_*/*.ckpt", "rank_*/*.save", "rank_*/*.save.tmp"):
            for save_file in Path(path).glob(pattern):
                if save_file.is_file():
                    save_file.unlink(missing_ok=True)

    @staticmethod
    def _read_checkpoint(checkpoint: Path) -> tuple[dict, int]:
        """
        Read one checkpoint file.

        Args:
            checkpoint: ``.ckpt`` log (one structure per line) or a legacy
                ``.save`` snapshot (one JSON dict) from older releases.

        Returns:
            Structures in the file and the number of damaged lines skipped.
        """
        if checkpoint.suffix == ".save":
            with open(checkpoint, "r") as chk:
                saved = json.load(chk)
            return {name: decode(xtal) for name, xtal in saved.items()}, 0

        structs, n_bad = {}, 0
        with open(checkpoint, "r") as chk:
            for line in chk:
                line = line.strip()
                if not line:
                    continue
                try:
                    name, xtal = line.split(":", 1)
                    structs[name.strip().strip('"')] = decode(
                        xtal.strip().rstrip(",")
                    )
                except (ValueError, KeyError, TypeError):
                    n_bad += 1
        return structs, n_bad

    def checkpoint_load(self, path: str, result_key: str) -> int:
        """
        Merges checkpoints of an interrupted run into the current structures
        and rebalances them across ranks. Unlike save, load is a collective
        and blocking operation.

        A checkpointed copy replaces the current copy of the same structure;
        structures that were never checkpointed (e.g. held by a rank that was
        killed before it finished its first calculation) are kept as they
        are. Checkpointed structures that are not in the current pool are
        ignored. Damaged lines and unreadable files are skipped with a
        warning; the affected structures are simply recomputed.

        Args:
            path: Directory containing rank_* checkpoint folders
            result_key: ``Atoms.info`` key that marks a structure as done

        Returns:
            Number of distinct structures restored from checkpoints.
        """
        # Legacy snapshots first (oldest to newest), then append-only logs,
        # so that later entries always hold the most complete copy
        checkpoints = None
        if gp.is_master:
            legacy = sorted(
                Path(path).glob("rank_*/*.save"), key=lambda p: p.stat().st_mtime
            )
            checkpoints = legacy + sorted(Path(path).glob("rank_*/*.ckpt"))
        checkpoints = gp.comm.bcast(checkpoints, root=0)
        if not checkpoints:
            # Nothing to merge: keep every rank's structures where they are
            self._checkpointed = {
                name
                for name, xtal in self.structs.items()
                if result_key in xtal.info
            }
            return 0

        # Each rank reads its share of the files
        restored, problems = [], []
        for idx in range(gp.rank, len(checkpoints), gp.size):
            checkpoint = checkpoints[idx]
            try:
                structs, n_bad = self._read_checkpoint(checkpoint)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f"{checkpoint} could not be read ({exc})")
                continue
            if n_bad:
                problems.append(f"{checkpoint}: {n_bad} damaged line(s) skipped")
            restored.extend((idx, name, xtal) for name, xtal in structs.items())

        restored = gp.comm.gather(restored, root=0)
        problems = gp.comm.gather(problems, root=0)
        current = gp.comm.gather(self.structs or {}, root=0)

        combined = None
        n_restored = 0
        if gp.is_master:
            for problem in chain.from_iterable(problems):
                self.logger.error(f"Checkpoint {problem}")
                gout.emit(
                    f"WARNING: Checkpoint {problem}. The affected structures "
                    "will be recomputed."
                )
            combined = {}
            for struct_dict in current:
                combined.update(struct_dict)
            checkpointed = {}
            for _, name, xtal in sorted(
                chain.from_iterable(restored), key=lambda item: item[0]
            ):
                if name in combined:
                    checkpointed[name] = xtal
            combined.update(checkpointed)
            n_restored = len(checkpointed)
            self.logger.debug(f"Read {n_restored} structures from checkpoints")

        self._scatter(combined)
        # Everything already completed is on disk; only new results get logged
        self._checkpointed = {
            name for name, xtal in self.structs.items() if result_key in xtal.info
        }
        return gp.comm.bcast(n_restored, root=0)

    def redistribute(self) -> None:
        """
        Redistribute structures such that all ranks have almost
        equal number of structures. Helpful for balancing load
        across cores
        """
        allstructs = gp.comm.gather(self.structs, root=0)
        combined_structs = None

        if gp.is_master:
            combined_structs = {}
            for struct_dict in allstructs:
                combined_structs.update(struct_dict)

        self._scatter(combined_structs)

    def _scatter(self, combined_structs: dict | None) -> None:
        """
        Split a structure dictionary evenly and scatter it to all ranks.

        Args:
            combined_structs: All structures; only read on the master rank.
        """
        scatter_list = None

        # Assemble the list to be scattered
        if gp.is_master:
            # Split dict into list of dicts
            items = list(combined_structs.items())
            num_per_rank = len(combined_structs) // gp.size
            remainder = len(combined_structs) % gp.size
            
            scatter_list = []
            start_idx = 0
            
            for rank in range(gp.size):
                slice_size = num_per_rank + (1 if rank < remainder else 0)
                end_idx = start_idx + slice_size
                scatter_list.append(dict(items[start_idx:end_idx]))
                start_idx = end_idx
                
            scatter_list.reverse()

        self.structs = gp.comm.scatter(scatter_list, root=0)
