Dataset
=======

Vortex files implement the Arrow Dataset interface permitting efficient use of a Vortex file within
query engines like DuckDB and Polars. In particular, Vortex will read data proportional to the
number of rows passing a filter condition and the number of columns in a selection. For most Vortex
encodings, this property holds true even when the filter condition specifies a single row.

A dataset is backed either by a single file, from :meth:`vortex.VortexFile.to_dataset`, or by many
files scanned as one table, from :meth:`vortex.VortexFiles.to_dataset`. For a multi-file dataset,
each file appears as one :class:`.VortexFragment`.

.. autosummary::
   :nosignatures:

   ~vortex.dataset.VortexDataset
   ~vortex.dataset.VortexScanner
   ~vortex.dataset.VortexFragment

.. raw:: html

   <hr>

.. automodule:: vortex.dataset
    :members:
