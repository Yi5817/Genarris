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
: Paths to conformer geometry files.
```{note}
  Any format supported by [`ase.io.read()`](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) works.
```

`Z` : `int`.
: Number of molecules per unit cell.

`log_level` : `str` | default = `info`.
: Python logging level (`debug`, `info`, `warning`, `error`).
