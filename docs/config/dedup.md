# Deduplication

Duplicate removal using
[`pymatgen StructureMatcher`](https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher).

```ini
[dedup]
stol        = 0.5
ltol        = 0.5
angle_tol   = 10
energy_key  = None
group_by_spg = True
```

`stol` : `float` | default = `0.5`.
: Site tolerance for `StructureMatcher`.

`ltol` : `float` | default = `0.5`.
: Fractional length tolerance for `StructureMatcher`.

`angle_tol` : `float` | default = `10`.
: Angle tolerance in degrees for `StructureMatcher`.

`energy_key` : `str` | default = `None`.
: If set, keep the lowest-energy structure among duplicates using this key
  from `xtal.info`.

`group_by_spg` : `bool` | default = `True`.
: Whether to group structures by space group. If `False`, all structures are
  deduplicated in one group.
