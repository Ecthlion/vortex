# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

import polars as pl
import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve
from vortex.polars_ import decompose_predicate, polars_to_vortex


@pytest.mark.parametrize(
    "polars, vortex",
    [
        (pl.col("AdvEngineID") != 0, ve.column("AdvEngineID") != 0),
        (pl.col("MobilePhoneModel") != "", ve.column("MobilePhoneModel") != ""),
        (pl.col("UserID") == 435090932899640449, ve.column("UserID") == 435090932899640449),
        # (pl.col("URL").str.contains("google"), ve.column("URL").str.contains("google")),
        # (
        #     (
        #         (pl.col("Title").str.contains("Google"))
        #         & (~pl.col("URL").str.contains(".google."))
        #         & (pl.col("SearchPhrase") != "")
        #     ),
        #     (
        #         (ve.column("Title").str.contains("Google"))
        #         & (~ve.column("URL").str.contains(".google."))
        #         & (ve.column("SearchPhrase") != "")
        #     ),
        # ),
        (pl.col("c") > 10000, ve.column("c") > 10000),
        #        (pl.col("EventDate") >= date(2013, 7, 1), ve.column("EventDate") >= date(2013, 7, 1)),
    ],
)
def test_exprs(polars: pl.Expr, vortex: ve.Expr) -> None:
    assert polars_to_vortex(polars).serialize() == vortex.serialize()


@pytest.fixture(scope="module")
def vxf(tmpdir_factory) -> vx.VortexFile:
    fname = tmpdir_factory.mktemp("data") / "polars_test.vortex"

    if not os.path.exists(fname):
        a = pa.array([{"index": x, "value": math.sqrt(x)} for x in range(1_000_000)])
        vx.io.write(vx.compress(vx.array(a)), str(fname))
    return vx.open(str(fname), without_segment_cache=True)


def test_to_polars_with_limit(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().limit(100).collect()
    assert len(df) == 100


def test_to_polars_with_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().filter(pl.col("index") < 500).collect()
    assert len(df) == 500
    assert df["index"].to_list() == list(range(500))


def test_to_polars_with_projection(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().select("index").limit(10).collect()
    assert df.columns == ["index"]
    assert len(df) == 10


def test_to_polars_with_projection_and_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().select("index", "value").filter(pl.col("index") < 100).collect()
    assert df.columns == ["index", "value"]
    assert len(df) == 100


def test_decompose_predicate_mixed() -> None:
    predicate = (pl.col("a") > 1) & pl.col("b").is_in([1, 2]) & (pl.col("c") == "x")
    pushed, residual = decompose_predicate(predicate)
    assert pushed == (ve.column("a") > 1) & (ve.column("c") == "x")
    assert residual is not None
    assert residual.meta.eq(pl.col("b").is_in([1, 2]))


def test_decompose_predicate_all_pushed() -> None:
    pushed, residual = decompose_predicate((pl.col("a") > 1) & (pl.col("b") == 2))
    assert pushed == (ve.column("a") > 1) & (ve.column("b") == 2)
    assert residual is None


def test_decompose_predicate_none_pushed() -> None:
    predicate = pl.col("a").is_in([1, 2])
    pushed, residual = decompose_predicate(predicate)
    assert pushed is None
    assert residual is not None
    assert residual.meta.eq(predicate)


def test_to_polars_with_unsupported_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().filter(pl.col("index").is_in([3, 7, 2_000_000])).collect()
    assert sorted(df["index"].to_list()) == [3, 7]


def test_to_polars_with_partially_supported_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().filter((pl.col("index") < 10) & pl.col("index").is_in([5, 7, 100])).collect()
    assert sorted(df["index"].to_list()) == [5, 7]


def test_to_polars_with_projection_and_unsupported_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().filter(pl.col("value").is_in([2.0, 3.0])).select("index").collect()
    assert df.columns == ["index"]
    assert sorted(df["index"].to_list()) == [4, 9]


def test_to_polars_with_unsupported_filter_and_limit(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().filter(pl.col("index").is_in(list(range(0, 1000, 2)))).limit(10).collect()
    assert df["index"].to_list() == list(range(0, 20, 2))
