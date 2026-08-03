# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import os
from typing import TYPE_CHECKING, final
from urllib.parse import urlparse
from urllib.request import url2pathname

import pyarrow as pa

from ._lib import files as _files
from ._lib.arrays import Array
from ._lib.dtype import DType
from ._lib.expr import Expr
from ._lib.iter import ArrayIterator
from .dataset import VortexDataset
from .file import VortexFile
from .file import open as open_file
from .type_aliases import IntoPaths, IntoProjection, IntoStore, RecordBatchReader

if TYPE_CHECKING:
    import polars

_GLOB_CHARS = "*?["


def _normalize_paths(paths: IntoPaths) -> str | list[str]:
    if isinstance(paths, (str, os.PathLike)):
        return os.fspath(paths)
    return [os.fspath(p) for p in paths]


def _names_many(path: str, *, glob: bool, local: bool) -> bool:
    """Whether `path` names many files (a directory or a glob pattern) rather than one file.

    Remote paths cannot be probed, so a remote directory is only recognized by its trailing
    slash. `local` is False when an explicit store makes paths store-relative.
    """
    if glob and any(c in path for c in _GLOB_CHARS):
        return True
    if path.endswith("/"):
        return True
    if not local:
        return False

    parsed = urlparse(path)
    if parsed.scheme == "file":
        return os.path.isdir(url2pathname(parsed.path))
    # A single-letter scheme is a Windows drive prefix, not a URL scheme.
    if len(parsed.scheme) > 1:
        return False
    return os.path.isdir(path)


def _open_source(paths: IntoPaths, *, store: IntoStore, glob: bool) -> VortexFile | VortexFiles:
    """Open `paths` as a single :class:`.VortexFile` or a multi-file :class:`.VortexFiles`."""
    paths = _normalize_paths(paths)
    if not isinstance(paths, str):
        return open_files(paths, store=store)
    if _names_many(paths, glob=glob, local=store is None):
        return open_files(paths, store=store)
    return open_file(paths, store=store)


def open_files(
    paths: IntoPaths,
    *,
    store: IntoStore = None,
) -> VortexFiles:
    """
    Lazily open many Vortex files as a single table.

    All files must share the same :class:`vortex.DType`.

    Parameters
    ----------
    paths : :class:`str` | :class:`os.PathLike` | Sequence[:class:`str` | :class:`os.PathLike`]
        A directory, a glob pattern, a single file, or a sequence of any of those. Local paths
        and URLs are both accepted. A directory - either an existing local directory or any path
        ending in ``/`` - is expanded to the ``*.vortex`` files it contains, recursively.
    store :
        An object store created from the `vortex.store` package. By default the store is inferred
        from each path. When given, paths are resolved relative to the store.

    Examples
    --------
    Read a directory of Vortex files into Polars:

    >>> import vortex as vx
    >>> lf = vx.open_files("nyc_taxi/").to_polars() # doctest: +SKIP

    Read only the files matching a glob pattern:

    >>> vxfs = vx.open_files("s3://bucket/year=2024/*.vortex") # doctest: +SKIP

    Read an explicit list of files:

    >>> vxfs = vx.open_files(["jan.vortex", "feb.vortex"]) # doctest: +SKIP

    See also: :func:`vortex.open`
    """
    return VortexFiles(_files.open_files(_normalize_paths(paths), store=store))


@final
class VortexFiles:
    """Many Vortex files scanned as a single table.

    Construct with :func:`vortex.open_files`.
    """

    def __init__(self, files: _files.VortexFiles):
        self._files = files

    @property
    def dtype(self) -> DType:
        """The dtype shared by every file."""
        return self._files.dtype

    @property
    def file_count(self) -> int:
        """The number of files backing this table."""
        return self._files.file_count

    def schema(self) -> pa.Schema:
        """The Arrow schema shared by every file."""
        return self._files.schema()

    def count_rows(self, *, expr: Expr | None = None) -> int:
        """Count the rows across all files, optionally keeping only rows matching ``expr``."""
        return self._files.count_rows(expr=expr)

    def scan(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        ordered: bool = True,
    ) -> ArrayIterator:
        """Scan the files, returning a :class:`vortex.ArrayIterator`.

        Parameters
        ----------
        projection : :class:`vortex.Expr` | list[str] | None
            The projection expression to read, or else read all columns.
        expr : :class:`vortex.Expr` | None
            The predicate used to filter rows. The filter columns do not need to be in the
            projection.
        limit : :class:`int` | None
            The maximum number of rows to read after filtering. If None, read all rows.
        ordered : :class:`bool`
            If ``True``, chunks are returned in file order. If ``False``, files are read
            concurrently and chunks may be interleaved.
        """
        return self._files.scan(projection, expr=expr, limit=limit, ordered=ordered)

    def to_arrow(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        schema: pa.Schema | None = None,
        ordered: bool = True,
    ) -> RecordBatchReader:
        """Scan the files as a :class:`pyarrow.RecordBatchReader`.

        Parameters
        ----------
        projection : :class:`vortex.Expr` | list[str] | None
            Either an expression over the columns of the files (only referenced columns will be
            read) or an explicit list of desired columns.
        expr : :class:`vortex.Expr` | None
            The predicate used to filter rows. The filter columns need not appear in the
            projection.
        limit : :class:`int` | None
            The maximum number of rows to read after filtering. If None, read all rows.
        schema : :class:`pyarrow.Schema` | None
            The Arrow schema to return. Use ``pyarrow.string()`` for ``StringArray`` fields.
            Use ``pyarrow.binary()`` for ``BinaryArray`` fields.
        ordered : :class:`bool`
            If ``True``, batches are returned in file order. If ``False``, files are read
            concurrently and batches may be interleaved.
        """
        return self._files.to_arrow(projection, expr=expr, limit=limit, schema=schema, ordered=ordered)

    def read_all(self, projection: IntoProjection = None, *, expr: Expr | None = None) -> Array:
        """Read every file into a single :class:`vortex.Array`."""
        return self.scan(projection, expr=expr).read_all()

    def to_dataset(self) -> VortexDataset:
        """Scan these files using the :class:`pyarrow.dataset.Dataset` API.

        The returned :class:`.VortexDataset` works anywhere a :class:`pyarrow.dataset.Dataset`
        does - DuckDB, Polars (``polars.scan_pyarrow_dataset``), pandas, and others - with
        column selection and row filters pushed into the scan. Each file appears as one
        :class:`.VortexFragment`.

        Examples
        --------
        >>> import duckdb
        >>> import vortex as vx
        >>> ds = vx.open_files("nyc_taxi/").to_dataset() # doctest: +SKIP
        >>> duckdb.sql("SELECT count(*) FROM ds WHERE passenger_count > 2") # doctest: +SKIP
        """
        return VortexDataset(self._files.to_dataset())

    def to_polars(self, *, ordered: bool = True) -> polars.LazyFrame:
        """Read the files as a ``polars.LazyFrame``, supporting column pruning and predicate pushdown.

        This is the Vortex equivalent of ``polars.scan_parquet("dir/")``.

        Parameters
        ----------
        ordered : :class:`bool`
            If ``True``, rows are produced in file order. If ``False``, files are read
            concurrently, which is faster but yields rows in a non-deterministic order.

        Examples
        --------
        >>> import polars as pl
        >>> import vortex as vx
        >>> lf = vx.open_files("nyc_taxi/").to_polars() # doctest: +SKIP
        >>> lf.filter(pl.col("passenger_count") > 2).select("fare_amount").collect() # doctest: +SKIP
        """
        from vortex.polars_ import lazy_frame

        def to_arrow(
            projection: IntoProjection = None,
            *,
            expr: Expr | None = None,
            limit: int | None = None,
        ) -> RecordBatchReader:
            return self.to_arrow(projection, expr=expr, limit=limit, ordered=ordered)

        return lazy_frame(to_arrow, self.dtype.to_arrow_schema())


def open_dataset(
    paths: IntoPaths,
    *,
    store: IntoStore = None,
    glob: bool = True,
) -> VortexDataset:
    """Open one Vortex file, or many, as a :class:`pyarrow.dataset.Dataset`.

    Accepts everything :func:`vortex.open_files` does - a directory, a glob pattern, or a list
    of files - as well as a single file, and returns a :class:`.VortexDataset` usable from
    DuckDB, Polars (``polars.scan_pyarrow_dataset``), pandas, and other Arrow dataset consumers.

    A single file keeps the full dataset feature set, including :meth:`.VortexDataset.take` and
    row-range fragments; a multi-file dataset exposes one fragment per file.

    Parameters
    ----------
    paths : :class:`str` | :class:`os.PathLike` | Sequence[:class:`str` | :class:`os.PathLike`]
        A single file, a directory, a glob pattern, or a sequence of any of those. Local paths
        and URLs are both accepted. A remote directory must end in ``/`` to be recognized.
    store :
        An object store created from the `vortex.store` package. By default the store is inferred
        from each path. When given, paths are resolved relative to the store.
    glob : :class:`bool`
        If ``False``, a path is never interpreted as a glob pattern. Use this to open a file
        whose name contains ``*``, ``?`` or ``[``.

    Examples
    --------
    >>> import duckdb
    >>> import vortex as vx
    >>> ds = vx.open_dataset("nyc_taxi/") # doctest: +SKIP
    >>> duckdb.sql("SELECT count(*) FROM ds WHERE passenger_count > 2") # doctest: +SKIP

    See also: :func:`vortex.scan_polars`
    """
    return _open_source(paths, store=store, glob=glob).to_dataset()


def scan_polars(
    paths: IntoPaths,
    *,
    store: IntoStore = None,
    glob: bool = True,
    ordered: bool = True,
) -> polars.LazyFrame:
    """Lazily scan one Vortex file, or many, as a ``polars.LazyFrame``.

    The Vortex equivalent of ``polars.scan_parquet``: accepts a single file, a directory, a
    glob pattern, or a list of files, and returns a ``polars.LazyFrame`` with column pruning
    and predicate pushdown.

    Parameters
    ----------
    paths : :class:`str` | :class:`os.PathLike` | Sequence[:class:`str` | :class:`os.PathLike`]
        A single file, a directory, a glob pattern, or a sequence of any of those. Local paths
        and URLs are both accepted. A remote directory must end in ``/`` to be recognized.
    store :
        An object store created from the `vortex.store` package. By default the store is inferred
        from each path. When given, paths are resolved relative to the store.
    glob : :class:`bool`
        If ``False``, a path is never interpreted as a glob pattern. Use this to open a file
        whose name contains ``*``, ``?`` or ``[``.
    ordered : :class:`bool`
        If ``True``, rows are produced in file order, sorted by path. If ``False``, files are
        read concurrently, which is faster but yields rows in a non-deterministic order.
        Ignored for a single file, whose rows are always in file order.

    Examples
    --------
    >>> import polars as pl
    >>> import vortex as vx
    >>> lf = vx.scan_polars("nyc_taxi/") # doctest: +SKIP
    >>> lf.filter(pl.col("passenger_count") > 2).select("fare_amount").collect() # doctest: +SKIP

    See also: :meth:`.VortexFile.to_polars`, :meth:`.VortexFiles.to_polars`
    """
    source = _open_source(paths, store=store, glob=glob)
    if isinstance(source, VortexFiles):
        return source.to_polars(ordered=ordered)
    return source.to_polars()
