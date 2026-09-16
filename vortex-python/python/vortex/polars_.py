# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
from collections.abc import Callable
from typing import Any, Literal

import polars as pl
import pyarrow as pa

import vortex.expr as ve

from ._lib import dtype as _dtype


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    data = json.loads(expr.meta.serialize(format="json"))
    assert isinstance(data, dict)
    return _polars_to_vortex(data)


_OPS: dict[str, Callable[[ve.Expr, ve.Expr], ve.Expr]] = {
    "Eq": ve.eq,
    "NotEq": ve.not_eq,
    "Lt": ve.lt,
    "LtEq": ve.lt_eq,
    "Gt": ve.gt,
    "GtEq": ve.gt_eq,
    "And": ve.and_,
    "Or": ve.or_,
    "LogicalAnd": ve.and_,
    "LogicalOr": ve.or_,
    "Plus": ve.add,
    "Minus": ve.sub,
    "Multiply": ve.mul,
    "TrueDivide": ve.div,
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
    "Float": lambda v: _dtype.float_(64, nullable=v is None),
    "Float32": lambda v: _dtype.float_(32, nullable=v is None),
    "Float64": lambda v: _dtype.float_(64, nullable=v is None),
    "Null": lambda v: _dtype.null(),
    "String": lambda v: _dtype.utf8(nullable=v is None),
    "Binary": lambda v: _dtype.binary(nullable=v is None),
    # Polars stores dates as days since the epoch, and times as nanoseconds since midnight.
    "Date": lambda v: _dtype.date("days", nullable=v is None),
    "Time": lambda v: _dtype.time("ns", nullable=v is None),
}


_TIME_UNITS: dict[str, Literal["s", "ms", "us", "ns"]] = {
    "Nanoseconds": "ns",
    "Microseconds": "us",
    "Milliseconds": "ms",
    "Seconds": "s",
}


def _timezone(tz: object) -> str | None:
    """Normalize a serialized Polars timezone, which newer versions wrap as ``{"inner": tz}``."""
    if tz is None:
        return None
    if isinstance(tz, str):
        return tz
    if isinstance(tz, dict):
        inner = tz.get("inner")
        if isinstance(inner, str):
            return inner
    raise NotImplementedError(f"Unsupported Polars timezone: {tz}")


def _timestamp_literal(value: int | None, unit: object, tz: object) -> ve.Expr:
    if not isinstance(unit, str) or unit not in _TIME_UNITS:
        raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")
    return ve.literal(_dtype.timestamp(_TIME_UNITS[unit], tz=_timezone(tz), nullable=value is None), value)


def _scalar_to_vortex(scalar: dict[str, Any]) -> ve.Expr:
    """Convert a serialized Polars ``Scalar`` to a Vortex literal expression."""
    scalar_type = next(iter(scalar.keys()), None)
    if scalar_type is None:
        raise NotImplementedError(f"Cannot convert to Vortex: empty Polars scalar {scalar}")
    value = scalar[scalar_type]

    if scalar_type == "Null":
        return ve.literal(_dtype.null(), None)

    if scalar_type in ("Datetime", "DateTime"):
        (value, unit, tz) = value
        return _timestamp_literal(value, unit, tz)

    if scalar_type == "Duration":
        raise NotImplementedError("Vortex has no duration type to represent a Polars Duration literal")

    if scalar_type == "Decimal":
        (value, precision, scale) = value
        return ve.literal(_dtype.decimal(precision=precision, scale=scale, nullable=value is None), value)

    if scalar_type == "Binary":
        return ve.literal(_dtype.binary(nullable=value is None), bytes(value))

    if scalar_type in _LITERAL_TYPES:
        return ve.literal(_LITERAL_TYPES[scalar_type](value), value)

    raise NotImplementedError(f"Cannot convert to Vortex: unsupported Polars scalar value type {scalar}")


_SIMPLE_DTYPES: dict[str, Callable[[], _dtype.DType]] = {
    "Boolean": lambda: _dtype.bool_(nullable=True),
    "Int8": lambda: _dtype.int_(8, nullable=True),
    "Int16": lambda: _dtype.int_(16, nullable=True),
    "Int32": lambda: _dtype.int_(32, nullable=True),
    "Int64": lambda: _dtype.int_(64, nullable=True),
    "UInt8": lambda: _dtype.uint(8, nullable=True),
    "UInt16": lambda: _dtype.uint(16, nullable=True),
    "UInt32": lambda: _dtype.uint(32, nullable=True),
    "UInt64": lambda: _dtype.uint(64, nullable=True),
    "Float32": lambda: _dtype.float_(32, nullable=True),
    "Float64": lambda: _dtype.float_(64, nullable=True),
    "String": lambda: _dtype.utf8(nullable=True),
    "Binary": lambda: _dtype.binary(nullable=True),
    "Date": lambda: _dtype.date("days", nullable=True),
    "Time": lambda: _dtype.time("ns", nullable=True),
    "Null": _dtype.null,
}


def _polars_dtype_to_vortex(dtype: object) -> _dtype.DType:
    """Convert a serialized Polars data type to a Vortex data type.

    Polars columns are always nullable, so the resulting Vortex types are nullable too.
    """
    if isinstance(dtype, str) and dtype in _SIMPLE_DTYPES:
        return _SIMPLE_DTYPES[dtype]()

    if isinstance(dtype, dict):
        if "Datetime" in dtype:
            (unit, tz) = dtype["Datetime"]
            if not isinstance(unit, str) or unit not in _TIME_UNITS:
                raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")
            return _dtype.timestamp(_TIME_UNITS[unit], tz=_timezone(tz), nullable=True)
        if "Decimal" in dtype:
            (precision, scale) = dtype["Decimal"]
            # Polars leaves the precision unset for inferred decimals, where it uses the maximum.
            precision = 38 if precision is None else precision
            scale = 0 if scale is None else scale
            return _dtype.decimal(precision=precision, scale=scale, nullable=True)
        if "List" in dtype:
            return _dtype.list_(_polars_dtype_to_vortex(dtype["List"]), nullable=True)

    raise NotImplementedError(f"Unsupported Polars data type: {dtype}")


def _like_escape(value: str) -> str:
    """Escape LIKE wildcards so that `value` matches literally within a pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _string_literal(expr: dict[str, Any]) -> str:
    """Extract a Python string from a serialized Polars string literal expression."""
    literal = expr.get("Literal")
    if isinstance(literal, dict):
        scalar = literal.get("Scalar", literal)
        if isinstance(scalar, dict):
            value = scalar.get("String")
            if isinstance(value, str):
                return value
    raise NotImplementedError(f"Expected a Polars string literal, got: {expr}")


_IS_IN_TYPES = (
    pa.types.is_integer,
    pa.types.is_floating,
    pa.types.is_string,
    pa.types.is_large_string,
    pa.types.is_string_view,
    pa.types.is_binary,
    pa.types.is_large_binary,
    pa.types.is_binary_view,
)


def _is_in_to_vortex(child: ve.Expr, values_expr: dict[str, Any]) -> ve.Expr:
    """Convert a Polars ``is_in`` over a literal set into an OR of equalities."""
    literal = values_expr.get("Literal")
    scalar = literal.get("Scalar", literal) if isinstance(literal, dict) else None
    if not isinstance(scalar, dict) or "List" not in scalar:
        raise NotImplementedError(f"Unsupported Polars is_in values: {values_expr}")

    # Polars serializes the set of values as an Arrow IPC stream.
    table = pa.ipc.open_stream(bytes(scalar["List"])).read_all()
    if table.num_columns != 1:
        raise NotImplementedError(f"Unsupported Polars is_in values: {values_expr}")
    column = table.column(0)

    if len(column) == 0:
        return ve.literal(_dtype.bool_(), False)
    if not any(check(column.type) for check in _IS_IN_TYPES):
        raise NotImplementedError(f"Unsupported Polars is_in value type: {column.type}")

    values = column.to_pylist()
    if any(value is None for value in values):
        # A null in the set never compares equal, so an OR of equalities would drop the
        # `nulls_equal` semantics that Polars applies to it.
        raise NotImplementedError("Unsupported null value in Polars is_in values")

    equalities = ve.or_collect(ve.eq(child, value) for value in values)
    assert equalities is not None
    return equalities


def _function_to_vortex(expr: dict[str, Any]) -> ve.Expr:
    """Convert a serialized Polars function expression to a Vortex expression."""
    inputs: list[dict[str, Any]] = expr["input"]
    fn = expr["function"]

    if "Boolean" in fn:
        fn = fn["Boolean"]

        if fn == "IsNull":
            return ve.is_null(_polars_to_vortex(inputs[0]))
        if fn == "IsNotNull":
            return ve.is_not_null(_polars_to_vortex(inputs[0]))
        if fn == "Not":
            return ve.not_(_polars_to_vortex(inputs[0]))

        if isinstance(fn, dict) and "IsBetween" in fn:
            closed = fn["IsBetween"]["closed"]
            return ve.between(
                _polars_to_vortex(inputs[0]),
                _polars_to_vortex(inputs[1]),
                _polars_to_vortex(inputs[2]),
                lower_strict=closed in ("Right", "None"),
                upper_strict=closed in ("Left", "None"),
            )

        if isinstance(fn, dict) and "IsIn" in fn:
            if fn["IsIn"]["nulls_equal"]:
                raise NotImplementedError(f"Unsupported nulls_equal argument in fn {expr}")
            return _is_in_to_vortex(_polars_to_vortex(inputs[0]), inputs[1])

        raise NotImplementedError(f"Unsupported Polars boolean function: {fn}")

    if "StringExpr" in fn:
        fn = fn["StringExpr"]
        # `%` and `_` in the search string must match literally rather than as LIKE wildcards.
        if fn == "StartsWith":
            return ve.like(_polars_to_vortex(inputs[0]), _like_escape(_string_literal(inputs[1])) + "%")
        if fn == "EndsWith":
            return ve.like(_polars_to_vortex(inputs[0]), "%" + _like_escape(_string_literal(inputs[1])))

        if isinstance(fn, dict) and "Contains" in fn:
            if not fn["Contains"]["literal"]:
                raise NotImplementedError("Unsupported regex pattern in Polars StringExpr.Contains")
            return ve.like(_polars_to_vortex(inputs[0]), "%" + _like_escape(_string_literal(inputs[1])) + "%")

        raise NotImplementedError(f"Unsupported Polars string function: {fn}")

    raise NotImplementedError(f"Unsupported Polars function: {fn}")


def _polars_to_vortex(expr: dict[str, Any]) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    if "BinaryExpr" in expr:
        expr = expr["BinaryExpr"]
        lhs = _polars_to_vortex(expr["left"])
        rhs = _polars_to_vortex(expr["right"])
        op = expr["op"]

        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return _OPS[op](lhs, rhs)

    if "Column" in expr:
        return ve.column(expr["Column"])

    if "Cast" in expr:
        expr = expr["Cast"]
        child = _polars_to_vortex(expr["expr"])

        dtype = expr["dtype"]
        # Post https://github.com/pola-rs/polars/pull/21797 the target is a DataTypeExpr.
        if isinstance(dtype, dict) and "Literal" in dtype:
            dtype = dtype["Literal"]
        return ve.cast(child, _polars_dtype_to_vortex(dtype))

    # See https://github.com/pola-rs/polars/pull/21849
    if "Scalar" in expr:
        return _scalar_to_vortex(expr["Scalar"])

    if "Literal" in expr:
        expr = expr["Literal"]

        literal_type = next(iter(expr.keys()), None)

        if literal_type == "Scalar":
            return _scalar_to_vortex(expr["Scalar"])

        # Special-case Series
        if literal_type == "Series":
            raise ValueError

        # Special-case date-times
        # (pre https://github.com/pola-rs/polars/pull/21849)
        if literal_type in ("DateTime", "Datetime"):
            (value, unit, tz) = expr[literal_type]
            return _timestamp_literal(value, unit, tz)

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
        return _function_to_vortex(expr["Function"])

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")
