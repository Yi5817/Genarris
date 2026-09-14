"""
Shared fixtures.

Layout:
    install/   package smoke tests (import, extension, data files, CLI)
    restart/   single-process tests of checkpoints and the restart manager
    workflow/  end-to-end runs under mpirun (marked ``integration``)
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def mpi_free_env() -> dict[str, str]:
    """
    Environment for child processes that start MPI themselves.

    Importing mpi4py anywhere in the test session initializes MPI in the
    pytest process, which exports launcher variables at the C level. A child
    that inherits them either fails to start or tries to join this process's
    MPI job, so hand it an explicit copy without them.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OMPI_", "PMIX_"))
    }
