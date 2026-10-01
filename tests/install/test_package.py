"""
The package, its compiled extension and its packaged data files are usable
with only the core dependencies installed.
"""
from __future__ import annotations

from importlib import resources

import pytest
import yaml
from packaging.version import Version


def test_package_imports_with_version() -> None:
    import gnrs

    assert gnrs.__version__, "package metadata not found; install with pip"
    Version(gnrs.__version__)  # raises if the version string is malformed


def test_compiled_cgenarris_extension_loads() -> None:
    from gnrs.cgenarris import pygenarris_mpi

    assert hasattr(
        pygenarris_mpi, "mpi_generate_molecular_crystals_with_vdw_cutoff_matrix"
    )


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("gnrs.parser", "defaults.yaml"),
        ("gnrs.gnrsutil", "h_bond.yaml"),
        ("gnrs.gnrsutil", "pymove_model.yaml"),
    ],
)
def test_packaged_data_files_are_installed(package: str, filename: str) -> None:
    path = resources.files(package) / filename
    assert path.is_file(), f"{filename} missing from {package}; check package-data"
    with path.open() as yfile:
        assert yaml.safe_load(yfile)


def test_defaults_cover_core_tasks() -> None:
    with (resources.files("gnrs.parser") / "defaults.yaml").open() as yfile:
        defaults = yaml.safe_load(yfile)
    for section in ("master", "generation", "rigid_press", "symm_rigid_press"):
        assert section in defaults
