"""
Fixtures for the end-to-end tests, which launch Genarris under mpirun.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def run_gnrs(
    mpi_free_env: dict[str, str],
) -> Callable[..., subprocess.CompletedProcess]:
    """
    Launcher of ``gnrs -c ui.conf`` under mpirun in a run directory.
    """

    def run(workdir: Path, *flags: str, nproc: int = 2) -> subprocess.CompletedProcess:
        # CI runners may have fewer slots than the restart test's three ranks.
        cmd = [
            shutil.which("mpirun"),
            "--oversubscribe",
            "-np",
            str(nproc),
            sys.executable,
            "-m",
            "gnrs.cli",
        ]
        return subprocess.run(
            cmd + ["-c", "ui.conf", *flags],
            cwd=workdir,
            env=mpi_free_env,
            capture_output=True,
            text=True,
            timeout=900,
        )

    return run
