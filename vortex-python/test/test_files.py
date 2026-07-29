# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Scanning a directory of Vortex files as a single table.

See https://github.com/vortex-data/vortex/discussions/5687.
"""

from pathlib import Path

import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve

ROWS_PER_FILE = 100
FILE_COUNT = 3


@pytest.fixture(scope="module")
def directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory of three Vortex files holding 0..300 in ``index``."""
    directory = tmp_path_factory.mktemp("files")
    for file in range(FILE_COUNT):
        start = file * ROWS_PER_FILE
        table = pa.table(
            {
                "index": pa.array(range(start, start + ROWS_PER_FILE), type=pa.int64()),
                "name": pa.array([f"row-{i}" for i in range(start, start + ROWS_PER_FILE)]),
            }
        )
        vx.io.write(vx.array(table), str(directory / f"part-{file}.vortex"))
    return directory


@pytest.fixture(
    params=["directory", "trailing_slash", "glob", "single_char_glob", "list", "file_url"],
)
def files(request: pytest.FixtureRequest, directory: Path) -> vx.VortexFiles:
    """The same three files, reached through each supported kind of source."""
    kind: str = request.param
    match kind:
        case "directory":
            return vx.open_files(str(directory))
        case "trailing_slash":
            return vx.open_files(f"{directory}/")
        case "glob":
            return vx.open_files(f"{directory}/*.vortex")
        case "single_char_glob":
            return vx.open_files(f"{directory}/part-?.vortex")
        case "list":
            return vx.open_files([str(directory / f"part-{i}.vortex") for i in range(FILE_COUNT)])
        case "file_url":
            return vx.open_files(directory.as_uri())
        case unknown:
            raise ValueError(f"unhandled source kind: {unknown}")


def test_file_count(files: vx.VortexFiles):
    assert files.file_count == FILE_COUNT


def test_dtype_and_schema(files: vx.VortexFiles):
    assert files.schema().names == ["index", "name"]
    assert files.dtype.to_arrow_schema().names == ["index", "name"]


def test_count_rows(files: vx.VortexFiles):
    assert files.count_rows() == FILE_COUNT * ROWS_PER_FILE


def test_count_rows_with_filter(files: vx.VortexFiles):
    assert files.count_rows(expr=ve.column("index") < 150) == 150


def test_read_all(files: vx.VortexFiles):
    table = files.read_all().to_arrow_table()
    assert table.num_rows == FILE_COUNT * ROWS_PER_FILE
    assert table.column("index").to_pylist() == list(range(FILE_COUNT * ROWS_PER_FILE))


def test_to_arrow_with_projection_and_filter(files: vx.VortexFiles):
    table = files.to_arrow(["index"], expr=ve.column("index") >= 250).read_all()
    assert table.column_names == ["index"]
    assert table.column("index").to_pylist() == list(range(250, 300))


def test_to_arrow_limit_is_global(files: vx.VortexFiles):
    """A limit bounds the whole scan, not each file."""
    table = files.to_arrow(limit=150).read_all()
    assert table.num_rows == 150
    assert table.column("index").to_pylist() == list(range(150))


def test_to_arrow_limit_beyond_row_count(files: vx.VortexFiles):
    assert files.to_arrow(limit=10_000).read_all().num_rows == FILE_COUNT * ROWS_PER_FILE


def test_unordered_reads_every_row(files: vx.VortexFiles):
    table = files.to_arrow(["index"], ordered=False).read_all()
    assert table.column("index").sort().to_pylist() == list(range(FILE_COUNT * ROWS_PER_FILE))


def test_directory_expansion_is_recursive(tmp_path: Path):
    nested = tmp_path / "year=2024" / "month=01"
    nested.mkdir(parents=True)
    array = vx.array(pa.table({"index": pa.array([1, 2, 3], type=pa.int64())}))
    vx.io.write(array, str(tmp_path / "top.vortex"))
    vx.io.write(array, str(nested / "deep.vortex"))

    files = vx.open_files(f"{tmp_path}/")
    assert files.file_count == 2
    assert files.count_rows() == 6


def test_directory_expansion_ignores_other_extensions(tmp_path: Path):
    vx.io.write(vx.array(pa.table({"index": pa.array([1], type=pa.int64())})), str(tmp_path / "a.vortex"))
    _ = (tmp_path / "_SUCCESS").write_text("")

    assert vx.open_files(str(tmp_path)).file_count == 1


def test_single_file(directory: Path):
    files = vx.open_files(str(directory / "part-1.vortex"))
    assert files.file_count == 1
    assert files.count_rows() == ROWS_PER_FILE


def test_no_matching_files(tmp_path: Path):
    with pytest.raises(Exception, match="No files matched"):
        _ = vx.open_files(f"{tmp_path}/")


def test_missing_file(tmp_path: Path):
    with pytest.raises(Exception, match="No files matched"):
        _ = vx.open_files(str(tmp_path / "absent.vortex"))


def test_empty_list_of_paths():
    with pytest.raises(TypeError, match="at least one"):
        _ = vx.open_files([])


def test_mismatched_dtypes(tmp_path: Path):
    vx.io.write(vx.array(pa.table({"a": pa.array([1, 2])})), str(tmp_path / "a.vortex"))
    vx.io.write(vx.array(pa.table({"b": pa.array(["x"])})), str(tmp_path / "b.vortex"))
    with pytest.raises(Exception, match="dtype mismatch"):
        _ = vx.open_files(f"{tmp_path}/").read_all()
