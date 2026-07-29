# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve
from vortex.polars_ import polars_to_vortex


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


@pytest.fixture(scope="module")
def directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory of four Vortex files, together holding 0..4000 in ``index``."""
    directory = tmp_path_factory.mktemp("polars_dir")
    for part in range(4):
        start = part * 1_000
        a = pa.array([{"index": x, "value": math.sqrt(x)} for x in range(start, start + 1_000)])
        vx.io.write(vx.array(a), str(directory / f"part-{part}.vortex"))
    return directory


def test_scan_directory(directory: Path):
    """The equivalent of ``pl.scan_parquet("dir/")``: https://github.com/vortex-data/vortex/discussions/5687"""
    lf = vx.open_files(f"{directory}/").to_polars()
    assert lf.collect_schema().names() == ["index", "value"]

    df = lf.collect()
    assert len(df) == 4_000
    assert df["index"].to_list() == list(range(4_000))


def test_scan_directory_with_limit(directory: Path):
    assert len(vx.open_files(str(directory)).to_polars().limit(100).collect()) == 100


def test_scan_directory_with_filter(directory: Path):
    df = vx.open_files(str(directory)).to_polars().filter(pl.col("index") >= 3_500).collect()
    assert len(df) == 500
    assert df["index"].to_list() == list(range(3_500, 4_000))


def test_scan_directory_with_projection_and_filter(directory: Path):
    df = vx.open_files(str(directory)).to_polars().select("index").filter(pl.col("index") % 1_000 == 0).collect()
    assert df.columns == ["index"]
    assert df["index"].to_list() == [0, 1_000, 2_000, 3_000]


def test_scan_glob(directory: Path):
    df = vx.open_files(f"{directory}/part-[01].vortex").to_polars().collect()
    assert df["index"].to_list() == list(range(2_000))


class TestUnsupportedPredicate:
    """Predicates Vortex cannot represent are applied by Vortex's IO source, not dropped.

    Polars does not re-check a predicate that an IO source declined, so an unapplied predicate
    would silently return excluded rows.
    """

    def test_filter(self, vxf: vx.VortexFile):
        df = vxf.to_polars().filter(pl.col("index") % 250_000 == 0).collect()
        assert df["index"].to_list() == [0, 250_000, 500_000, 750_000]

    def test_filter_over_directory(self, directory: Path) -> None:
        df = vx.open_files(str(directory)).to_polars().filter(pl.col("index") % 1_000 == 0).collect()
        assert df["index"].to_list() == [0, 1_000, 2_000, 3_000]

    def test_projection_excluding_predicate_column(self, directory: Path) -> None:
        """The predicate column is read to filter with, then dropped from the result."""
        df = vx.open_files(str(directory)).to_polars().filter(pl.col("index") % 1_000 == 0).select("value").collect()
        assert df.columns == ["value"]
        assert df["value"].to_list() == [math.sqrt(i) for i in (0, 1_000, 2_000, 3_000)]

    def test_limit_is_applied_after_the_filter(self, directory: Path) -> None:
        df = vx.open_files(str(directory)).to_polars().filter(pl.col("index") % 1_000 == 0).limit(3).collect()
        assert df["index"].to_list() == [0, 1_000, 2_000]

    def test_no_matching_rows(self, directory: Path) -> None:
        df = vx.open_files(str(directory)).to_polars().filter(pl.col("index") % 10 == 11).collect()
        assert df.columns == ["index", "value"]
        assert len(df) == 0


def test_scan_directory_unordered(directory: Path):
    df = vx.open_files(str(directory)).to_polars(ordered=False).collect()
    assert sorted(df["index"].to_list()) == list(range(4_000))
