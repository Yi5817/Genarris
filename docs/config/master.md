# Master

Top-level project settings.

```ini
[master]
name            = aspirin
molecule_path   = ["./aspirin.xyz"]
Z               = 4
log_level       = info
```

`name` : `str`.
: Project name, used for output directories and log files.

`molecule_path` : `list[str]`.
: Paths to conformer geometry files. Give one file per molecule; several
  files are used by {doc}`asu_generation`.
```{note}
  Any format supported by [`ase.io.read()`](https://docs.ase-lib.org/ase/io/io.html) works.
```

`Z` : `int`.
: Number of molecules per unit cell. Not needed when the workflow only runs
  `asu_generation`.

`log_level` : `str` | default = `info`.
: Python logging level (`debug`, `info`, `warning`, `error`) of `Genarris.log`.
  Rank 0 writes the file. Warnings and errors of the other MPI ranks are added
  when a task ends, one line per distinct message, with the rank or the number
  of ranks that logged it.
