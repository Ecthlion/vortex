# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import json
import operator
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, cast

import polars as pl
import pyarrow as pa

import vortex.expr as ve

from ._lib import dtype as _dtype

if TYPE_CHECKING:
    from .type_aliases import IntoProjection, RecordBatchReader


class _ToArrow(Protocol):
    """The subset of the Vortex reader APIs that :func:`lazy_frame` scans through."""

    def __call__(
        self,
        projection: IntoProjection = None,
        *,
        expr: ve.Expr | None = None,
        limit: int | None = None,
    ) -> RecordBatchReader: ...


def lazy_frame(to_arrow: _ToArrow, schema: pa.Schema) -> pl.LazyFrame:
    """Register a Polars IO source that scans Vortex, pruning columns and pushing down predicates.

    Predicates that Vortex cannot represent are applied here instead of being pushed into the
    scan. Polars does not re-check a predicate the IO source declined, so leaving one unapplied
    would return rows the query excluded.

    Parameters
    ----------
    to_arrow :
        Called with the columns and predicate Polars pushed down, returning a
        :class:`pyarrow.RecordBatchReader`.
    schema : :class:`pyarrow.Schema`
        The schema of the unprojected source.
    """
    from polars.io.plugins import register_io_source

    def _io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        _batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # TODO(ngates): split a conjunction so the convertible terms can still be pushed down.
        pushdown, residual = _split_predicate(predicate)

        projection = with_columns
        if residual is not None and projection is not None:
            # The residual filter runs here, so the columns it reads must be scanned even when
            # the query does not select them.
            projection = list(dict.fromkeys([*projection, *residual.meta.root_names()]))

        # A limit may only be pushed down when the scan applies the whole predicate, otherwise it
        # would discard rows before the residual filter has run.
        reader = to_arrow(projection, expr=pushdown, limit=n_rows if residual is None else None)

        def to_frame(batch: pa.RecordBatch) -> pl.DataFrame:
            # TODO(ngates): set sortedness on DataFrame based on stats?
            df = pl.DataFrame._from_arrow(batch, rechunk=False)
            if residual is None:
                return df
            df = df.filter(residual)
            return df if with_columns is None else df.select(with_columns)

        remaining = n_rows
        for batch in reader:
            df = to_frame(batch)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= len(df)
            yield df
            if remaining == 0:
                return

        # Make sure we always yield at least one empty DataFrame
        yield to_frame(
            pa.RecordBatch.from_arrays(
                [pa.array([], type=field.type) for field in reader.schema],
                schema=reader.schema,
            )
        )

    # https://github.com/pola-rs/polars/pull/24125
    return register_io_source(_io_source, schema=schema)  # ty: ignore[invalid-argument-type]


def _split_predicate(predicate: pl.Expr | None) -> tuple[ve.Expr | None, pl.Expr | None]:
    """Split a pushed-down predicate into the part Vortex evaluates and the part Polars does.

    Returns ``(pushdown, residual)``, exactly one of which is set when ``predicate`` is given.
    """
    if predicate is None:
        return None, None
    try:
        return polars_to_vortex(predicate), None
    except (NotImplementedError, ValueError):
        return None, predicate


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    data = json.loads(expr.meta.serialize(format="json"))
    assert isinstance(data, dict)
    return _polars_to_vortex(data)


_OPS = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Lt": operator.lt,
    "LtEq": operator.le,
    "Gt": operator.gt,
    "GtEq": operator.ge,
    "And": operator.and_,
    "Or": operator.or_,
    "LogicalAnd": operator.and_,
    "LogicalOr": operator.or_,
}


_LITERAL_TYPES: dict[str, Callable[[Any | None], _dtype.DType]] = {
    "Boolean": lambda v: _dtype.bool_(nullable=v is None),
    "Int": lambda v: _dtype.int_(64, nullable=v is None),
    "Int8": lambda v: _dtype.int_(8, nullable=v is None),
    "Int16": lambda v: _dtype.int_(16, nullable=v is None),
    "Int32": lambda v: _dtype.int_(32, nullable=v is None),
    "Int64": lambda v: _dtype.int_(64, nullable=v is None),
    "UInt8": lambda v: _dtype.uint(8, nullable=v is None),
    "UInt16": lambda v: _dtype.uint(16, nullable=v is None),
    "UInt32": lambda v: _dtype.uint(32, nullable=v is None),
    "UInt64": lambda v: _dtype.uint(64, nullable=v is None),
    "Float32": lambda v: _dtype.float_(32, nullable=v is None),
    "Float64": lambda v: _dtype.float_(64, nullable=v is None),
    "Null": lambda v: _dtype.null(),
    "String": lambda v: _dtype.utf8(nullable=v is None),
    "Binary": lambda v: _dtype.binary(nullable=v is None),
}


def _polars_to_vortex(expr: dict[str, Any]) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    if "BinaryExpr" in expr:
        expr = expr["BinaryExpr"]
        lhs = _polars_to_vortex(expr["left"])
        rhs = _polars_to_vortex(expr["right"])
        op = expr["op"]

        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return cast(ve.Expr, _OPS[op](lhs, rhs))

    if "Column" in expr:
        return ve.column(expr["Column"])

    # See https://github.com/pola-rs/polars/pull/21849
    if "Scalar" in expr:
        scalar = expr["Scalar"]

        if "Null" in scalar:
            value = None
            dtype = "Null"
        elif "String" in scalar:
            value = scalar["String"]
            dtype = "String"
        elif "Int" in scalar:
            value = scalar["Int"]
            dtype = "Int64"
        elif "Float" in scalar:
            value = scalar["Float"]
            dtype = "Float64"
        elif "Float32" in scalar:
            value = scalar["Float32"]
            dtype = "Float32"
        elif "Float64" in scalar:
            value = scalar["Float64"]
            dtype = "Float64"
        elif "Int32" in scalar:
            value = scalar["Int32"]
            dtype = "Int32"
        elif "Int64" in scalar:
            value = scalar["Int64"]
            dtype = "Int64"
        else:
            raise ValueError(f"Cannot convert to Vortex: unsupported Polars scalar value type {scalar}")

        return ve.literal(_LITERAL_TYPES[dtype](value), value)

    if "Literal" in expr:
        expr = expr["Literal"]

        literal_type = next(iter(expr.keys()), None)

        if literal_type == "Scalar":
            return _polars_to_vortex(expr)

        # Special-case Series
        if literal_type == "Series":
            raise ValueError

        # Special-case date-times
        if literal_type == "DateTime":
            (value, unit, tz) = expr[literal_type]
            if unit == "Nanoseconds":
                unit = "ns"
            elif unit == "Microseconds":
                unit = "us"
            elif unit == "Milliseconds":
                unit = "ms"
            elif unit == "Seconds":
                unit = "s"
            else:
                raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")

            dtype = _dtype.timestamp(unit, tz=tz, nullable=value)
            return ve.literal(dtype, value)

        # Unwrap 'Dyn' scalars, whose type hasn't been established yet.
        # (post https://github.com/pola-rs/polars/pull/21849)
        if literal_type == "Dyn":
            expr = expr["Dyn"]
            literal_type = next(iter(expr.keys()), None)

        if literal_type not in _LITERAL_TYPES:
            raise NotImplementedError(f"Unsupported Polars literal type: {literal_type}")
        value = expr[literal_type]
        return ve.literal(_LITERAL_TYPES[literal_type](value), value)

    if "Function" in expr:
        expr = expr["Function"]
        _inputs = [_polars_to_vortex(e) for e in expr["input"]]

        fn = expr["function"]
        if "Boolean" in fn:
            fn = fn["Boolean"]

            if "IsIn" in fn:
                fn = fn["IsIn"]
                if fn["nulls_equal"]:
                    raise ValueError(f"Unsupported nulls_equal argument in fn {expr}")

                # Vortex doesn't support is-in, so we need to construct a series of ORs?

        if "StringExpr" in fn:
            fn = fn["StringExpr"]
            if "Contains" in fn:
                raise ValueError("Unsupported Polars StringExpr.Contains")

        raise NotImplementedError(f"Unsupported Polars function: {fn}")

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")
