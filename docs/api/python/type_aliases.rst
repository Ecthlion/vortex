Type Aliases
============

.. data:: vortex.type_aliases.IntoArrayIterator

          Anything that can produce a sequence of Vortex Arrays.


.. data:: vortex.type_aliases.IntoProjection

          An expression, a list of column names, or None.

          Only the data necessary to evaluate the expression or produce the explicit column list are read.

          If None, all columns from the file are read.

.. data:: vortex.type_aliases.IntoPaths

          A path, or a sequence of paths, each given as a string or a path-like object.

          A path may name a single file, a directory, or a glob pattern, as a local path or a URL.


.. data:: vortex.type_aliases.IntoStore

          An object store from the ``vortex.store`` package, or None to infer one from each path.
