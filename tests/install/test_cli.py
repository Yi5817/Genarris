"""
The ``gnrs`` command line entry point is installed and parses its options.
"""
from __future__ import annotations

import shutil
import subprocess


def test_console_script_is_installed(mpi_free_env: dict[str, str]) -> None:
    gnrs_cmd = shutil.which("gnrs")
    assert gnrs_cmd, "the gnrs command is not on PATH; install with pip"

    proc = subprocess.run(
        [gnrs_cmd, "--help"],
        env=mpi_free_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--config", "--seed", "--restart", "--overwrite"):
        assert flag in proc.stdout


def test_restart_and_overwrite_are_exclusive(mpi_free_env: dict[str, str]) -> None:
    proc = subprocess.run(
        [shutil.which("gnrs"), "-c", "ui.conf", "--restart", "--overwrite"],
        env=mpi_free_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "not allowed with" in proc.stderr
