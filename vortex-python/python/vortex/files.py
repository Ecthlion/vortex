# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, final

import pyarrow as pa

from ._lib import files as _files
from ._lib.arrays import Array
from ._lib.dtype import DType
from ._lib.expr import Expr
from ._lib.iter import ArrayIterator
from .dataset import VortexDataset
from .store import (
    AzureStore,
    GCSStore,
    HTTPStore,
    LocalStore,
    MemoryStore,
    S3Store,
)
from .type_aliases import IntoProjection, RecordBatchReader

if TYPE_CHECKING:
    import polars


def open_files(
    paths: str | Sequence[str],
    *,
    store: AzureStore | GCSStore | HTTPStore | LocalStore | MemoryStore | S3Store | None = None,
) -> VortexFiles:
    """
    Lazily open many Vortex files as a single table.

    All files must share the same :class:`vortex.DType`.

    Parameters
    ----------
    paths : :class:`str` | Sequence[:class:`str`]
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
    return VortexFiles(_files.open_files(paths, store=store))


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
