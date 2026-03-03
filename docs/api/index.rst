API Reference
=============

This section provides the full API reference for the ``gnrs`` package,
auto-generated from source code docstrings.

.. note::

   - All classes use **MPI communicators** (``mpi4py.MPI.Comm``) for parallel execution.
   - Crystal structures are represented as ``ase.Atoms`` objects with metadata in ``atoms.info``.
   - Structure collections are **distributed dictionaries** (``dict[str, Atoms]``) split across MPI ranks.

.. toctree::
   :maxdepth: 2

   gnrs.workflow
   gnrs.core
   gnrs.generation
   gnrs.optimize
   gnrs.energy
   gnrs.descriptor
   gnrs.cluster
   gnrs.deduplication
   gnrs.gnrsutil
   gnrs.parallel
