"""
Utility for building MPI-aware DFT execution commands.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
import subprocess

logger = logging.getLogger("mpi_cmd")

_VALID_LAUNCHERS = {"mpirun", "srun", "ibrun", "none"}


def build_dft_command(task_settings: dict, rank: int) -> str:
    """
    Build the command used by ASE to invoke a DFT binary.

    Args:
        task_settings: Energy/optimization settings
        rank: MPI rank of the calling process
    
    Returns:
        Command string to be passed to the ASE calculator
    """
    binary = task_settings["command"]
    launcher = task_settings.get("mpi_launcher", "mpirun")
    dft_mode = task_settings.get("dft_mode", "parallel")

    if launcher not in _VALID_LAUNCHERS:
        raise ValueError(
            f"Invalid mpi_launcher={launcher!r}."
        )

    if launcher == "none":
        logger.info("mpi_launcher=none: running DFT binary directly (no MPI wrapper)")
        return binary

    num_cores = task_settings["num_cores"]

    if launcher == "srun":
        return f"srun -n {num_cores} {binary}"
    
    elif launcher == "ibrun":
        return f"ibrun {binary}"

    # launcher == "mpirun"
    if dft_mode == "serial":
        return f"mpirun -np {num_cores} {binary}"

    if task_settings.get("use_slurm", False):
        host = _get_slurm_host(rank)
        return f"mpirun -host {host} -np {num_cores} {binary}"

    return f"mpirun -np {num_cores} {binary}"


def _get_slurm_host(rank: int) -> str:
    """
    Resolve the SLURM hostname for the given rank.

    Args:
        rank: MPI rank of the calling process
    
    Returns:
        Hostname string for the target node
    """
    cmd = "scontrol show hostname $SLURM_JOB_NODELIST"
    all_hosts = (
        subprocess.check_output(cmd, shell=True)
        .decode()
        .strip()
        .split("\n")
    )
    return all_hosts[rank + 1]
