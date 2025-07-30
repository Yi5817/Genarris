import os
import sys
import importlib
from subprocess import run, PIPE
import sysconfig
from setuptools import find_packages, setup, Extension

# Set location of MPI C compiler (mpicc) here:
# To detect automatically from environment, leave it as None
MPI_C_COMPILER = None


def test_py_version():
    py_min_version = (3, 8)
    if sys.version_info < py_min_version:
        raise SystemExit(
            f"Genarris installation requires Python {'.'.join(map(str, py_min_version))} or higher."
        )


def set_mpicc():
    global MPI_C_COMPILER
    out = run(["which", "mpicc"], stdout=PIPE)
    if out.returncode != 0 and MPI_C_COMPILER is None:
        raise SystemExit(
            "Unable to detect MPI C compiler (mpicc). Please ensure MPI is installed."
        )
    else:
        # Use detected MPI compiler if none was specified
        if MPI_C_COMPILER is None:
            MPI_C_COMPILER = out.stdout.decode("ascii").strip()

        print(f"Using MPI C compiler: {MPI_C_COMPILER}")
        print(
            "NOTE: It is recommended that all MPI packages "
            "are installed using the same compiler."
        )

        # Configure compiler variables
        ccvars = sysconfig.get_config_vars()
        for key in ["CC", "LDSHARED"]:
            if key in ccvars:
                value = ccvars[key].split()
                value[0] = MPI_C_COMPILER
                ccvars[key] = " ".join(value)
                print(f"Using {key} flags: {ccvars[key]}")


# Cgenarris Extension (aka pygenarris)
def get_pygenarris_sources():
    src_spglib = [
        "arithmetic.c",
        "cell.c",
        "delaunay.c",
        "determination.c",
        "hall_symbol.c",
        "kgrid.c",
        "kpoint.c",
        "mathfunc.c",
        "niggli.c",
        "overlap.c",
        "pointgroup.c",
        "primitive.c",
        "refinement.c",
        "sitesym_database.c",
        "site_symmetry.c",
        "spacegroup.c",
        "spin.c",
        "spg_database.c",
        "spglib.c",
        "symmetry.c",
    ]

    src_pygenarris = [
        "algebra.c",
        "check_structure.c",
        "combinatorics.c",
        "crystal_utils.c",
        "lattice_generator.c",
        "molecule_placement.c",
        "molecule_utils.c",
        "pygenarris_mpi.c",
        "pygenarris_mpi.i",
        "pygenarris_mpi_utils.c",
        "randomgen.c",
        "read_input.c",
        "spg_generation.c",
    ]

    pygenarris_src_dir = "./gnrs/cgenarris/src/"
    spglib_src_dir = os.path.join(pygenarris_src_dir, "spglib_src")

    if not os.path.exists(pygenarris_src_dir):
        raise SystemExit(
            "Unable to find genarris C extension. Please initialize the cgenarris submodule."
        )

    # Create __init__.py files if missing
    inits = [
        os.path.join(pygenarris_src_dir, "__init__.py"),
        os.path.join(pygenarris_src_dir, "..", "__init__.py"),
    ]
    for init in inits:
        if not os.path.isfile(init):
            open(init, "a").close()

    sources = []
    for src in src_pygenarris:
        sources.append(os.path.join(pygenarris_src_dir, src))
    for src in src_spglib:
        sources.append(os.path.join(spglib_src_dir, src))

    include = [pygenarris_src_dir, spglib_src_dir]

    # Get numpy and mpi4py include
    required_mods = ["numpy", "mpi4py"]
    for mod in required_mods:
        try:
            loaded_mod = importlib.import_module(mod)
            include.append(getattr(loaded_mod, "get_include")())
        except ModuleNotFoundError:
            raise SystemExit(f"Please install {mod} before installing Genarris!")

    return sources, include

# Rigid_press C Extension (aka rpack)
def get_rigid_press_sources():
    src_rpress = [
        "rigid_press.i",
        "rigid_press.c",
        "symmetrization.c",
        "d_algebra.c",
    ]

    src_spglib = [
        "arithmetic.c",
        "cell.c",
        "delaunay.c",
        "determination.c",
        "hall_symbol.c",
        "kgrid.c",
        "kpoint.c",
        "mathfunc.c",
        "niggli.c",
        "overlap.c",
        "pointgroup.c",
        "primitive.c",
        "refinement.c",
        "sitesym_database.c",
        "site_symmetry.c",
        "spacegroup.c",
        "spin.c",
        "spg_database.c",
        "spglib.c",
        "symmetry.c",
    ]

    rpress_source_dir = "./gnrs/cgenarris/src/rpack/rigid_press"
    spglib_src_dir = "./gnrs/cgenarris/src/spglib_src"

    sources = []
    for src in src_rpress:
        sources.append(os.path.join(rpress_source_dir, src))
    for src in src_spglib:
        sources.append(os.path.join(spglib_src_dir, src))

    include_rpress = [rpress_source_dir, spglib_src_dir]

    required_mods = ["numpy"]
    for mod in required_mods:
        try:
            loaded_mod = importlib.import_module(mod)
            include_rpress.append(getattr(loaded_mod, "get_include")())
        except ModuleNotFoundError:
            raise SystemExit(f"Please install {mod} before installing Genarris!")

    return sources, include_rpress

test_py_version()
set_mpicc()
sources_pygenarris, include_pygenarris = get_pygenarris_sources()
pygenarris_mpi = Extension(
    "gnrs.cgenarris.src._pygenarris_mpi",
    include_dirs=include_pygenarris,
    sources=sources_pygenarris,
    extra_compile_args=["-std=gnu99", "-O3"],
)

# Add rigid_press extension
sources_rigid_press, include_rigid_press = get_rigid_press_sources()
rigid_press = Extension(
    "gnrs.cgenarris.src.rpack.rigid_press._rigid_press",
    include_dirs=include_rigid_press,
    sources=sources_rigid_press,
    extra_compile_args=["-std=gnu99", "-O3"],
    libraries=["lapack", "blas"],
)

setup(
    ext_modules=[pygenarris_mpi, rigid_press],
    py_modules=["pygenarris_mpi", "rigid_press"],
    zip_safe=False
)
