"""
End-to-end restart tests. Each test launches Genarris under mpirun on a tiny
benzene workflow (generation + symm_rigid_press), so they take a few minutes.

Run with:  pytest -m integration
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MPIRUN = shutil.which("mpirun")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(MPIRUN is None, reason="mpirun not found"),
]

BENZENE_XYZ = """12
benzene
C   0.000   1.396   0.000
C   1.209   0.698   0.000
C   1.209  -0.698   0.000
C   0.000  -1.396   0.000
C  -1.209  -0.698   0.000
C  -1.209   0.698   0.000
H   0.000   2.479   0.000
H   2.147   1.240   0.000
H   2.147  -1.240   0.000
H   0.000  -2.479   0.000
H  -2.147  -1.240   0.000
H  -2.147   1.240   0.000
"""

TASKS = "['generation', 'symm_rigid_press']"

CONFIG = f"""[master]
name = benzene
molecule_path = ["./benzene.xyz"]
z = 2
log_level = info

[workflow]
tasks = {TASKS}

[generation]
num_structures_per_spg = 3
spg_distribution_type = [2, 14]
sr = 0.95
tol = 0.01
ucv_mean = predict
ucv_mult = 1.5
natural_cutoff_mult = 1.2

[symm_rigid_press]
sr = 0.85
method = BFGS
tol = 0.01
natural_cutoff_mult = 1.2
maxiter = 100
"""


def run_gnrs(
    workdir: Path, env: dict[str, str], *flags: str, nproc: int = 2
) -> subprocess.CompletedProcess:
    cmd = [MPIRUN, "-np", str(nproc), sys.executable, "-m", "gnrs.cli", "-c", "ui.conf"]
    return subprocess.run(
        cmd + list(flags),
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def flat(text: str) -> str:
    """
    Collapse whitespace so line-wrapped console messages can be matched.
    """
    return " ".join(text.split())


def n_structures(path: Path) -> int:
    with open(path) as sfile:
        return len(json.load(sfile))


def make_interrupted(workdir: Path) -> None:
    """
    Turn a finished run into one that was killed during symm_rigid_press:
    drop the task from the restart record and delete its output, keeping the
    per-rank checkpoint logs.
    """
    restart_file = workdir / "restart.json"
    restart = json.loads(restart_file.read_text())
    restart["gnrs_info"].pop("symm_rigid_press")
    restart["gnrs_info"]["last_struct_path"] = restart["gnrs_info"]["generation"]["results"]
    restart_file.write_text(json.dumps(restart))
    (workdir / "structures" / "symm_rigid_press" / "structures.json").unlink()


@pytest.fixture(scope="module")
def finished_run(
    tmp_path_factory: pytest.TempPathFactory, mpi_free_env: dict[str, str]
) -> Path:
    workdir = tmp_path_factory.mktemp("finished")
    (workdir / "benzene.xyz").write_text(BENZENE_XYZ)
    (workdir / "ui.conf").write_text(CONFIG)
    proc = run_gnrs(workdir, mpi_free_env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "All tasks completed successfully" in proc.stdout
    assert (workdir / "restart.json").is_file()
    return workdir


@pytest.fixture
def run_copy(finished_run: Path, tmp_path: Path) -> Path:
    workdir = tmp_path / "run"
    shutil.copytree(finished_run, workdir)
    return workdir


def test_resume_interrupted_task_on_other_process_count(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    make_interrupted(run_copy)
    logs = sorted((run_copy / "tmp" / "symm_rigid_press").glob("rank_*/*.ckpt"))
    assert logs, "the finished run should have left checkpoint logs"
    # One log cut short by the job kill, the others never written
    text = logs[0].read_text()
    logs[0].write_text(text[: len(text) // 2])
    for log in logs[1:]:
        log.unlink()

    proc = run_gnrs(run_copy, mpi_free_env, "--restart", nproc=1)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = flat(proc.stdout)
    assert "Completed tasks (will be skipped): generation" in out
    assert "Resuming from task: symm_rigid_press" in out
    assert "damaged line(s) skipped" in out
    assert "Checkpoints from a previous run found" in out
    n_in = n_structures(run_copy / "structures" / "generation" / "structures.json")
    n_out = n_structures(run_copy / "structures" / "symm_rigid_press" / "structures.json")
    assert n_out == n_in, "no structure may be lost on restart"


def test_restart_of_finished_run_does_nothing(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    proc = run_gnrs(run_copy, mpi_free_env, "--restart")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "All tasks were already completed. Nothing to do." in proc.stdout


def test_restart_without_previous_run_is_refused(
    tmp_path: Path, mpi_free_env: dict[str, str]
) -> None:
    (tmp_path / "benzene.xyz").write_text(BENZENE_XYZ)
    (tmp_path / "ui.conf").write_text(CONFIG)
    proc = run_gnrs(tmp_path, mpi_free_env, "--restart")
    assert proc.returncode != 0
    assert "no restart file was found" in flat(proc.stdout)


def test_restart_with_changed_task_list_is_refused(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    conf = run_copy / "ui.conf"
    conf.write_text(
        conf.read_text().replace(TASKS, "['generation', 'dedup', 'symm_rigid_press']")
    )
    proc = run_gnrs(run_copy, mpi_free_env, "--restart")
    assert proc.returncode != 0
    assert "task list changed" in flat(proc.stdout)


def test_restart_with_changed_completed_settings_is_refused(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    make_interrupted(run_copy)
    conf = run_copy / "ui.conf"
    conf.write_text(conf.read_text().replace("sr = 0.95", "sr = 0.90"))
    proc = run_gnrs(run_copy, mpi_free_env, "--restart")
    assert proc.returncode != 0
    assert "generation.sr: 0.95 -> 0.9" in flat(proc.stdout)


def test_restart_with_changed_pending_settings_warns(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    make_interrupted(run_copy)
    conf = run_copy / "ui.conf"
    conf.write_text(conf.read_text().replace("sr = 0.85", "sr = 0.80"))
    proc = run_gnrs(run_copy, mpi_free_env, "--restart")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "symm_rigid_press.sr: 0.85 -> 0.8" in flat(proc.stdout)


def test_fresh_run_over_previous_run_is_refused(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    proc = run_gnrs(run_copy, mpi_free_env)
    assert proc.returncode != 0
    assert "--overwrite" in proc.stdout
    assert (run_copy / "restart.json").is_file(), "nothing may be touched"


def test_fresh_run_over_old_layout_run_is_refused(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    # Older releases kept the restart file under tmp/
    old_file = run_copy / "tmp" / "restart.json"
    (run_copy / "restart.json").rename(old_file)
    proc = run_gnrs(run_copy, mpi_free_env)
    assert proc.returncode != 0
    assert "--overwrite" in proc.stdout
    assert old_file.is_file(), "nothing may be touched"


def test_overwrite_starts_over(
    run_copy: Path, mpi_free_env: dict[str, str]
) -> None:
    # A checkpoint of a task the new run never reaches must not survive
    stale = run_copy / "tmp" / "never_run" / "rank_0"
    stale.mkdir(parents=True)
    (stale / "0.ckpt").write_text("")
    proc = run_gnrs(run_copy, mpi_free_env, "--overwrite")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Discarding the previous run's progress record" in flat(proc.stdout)
    assert "All tasks completed successfully" in proc.stdout
    assert not (stale / "0.ckpt").exists()
