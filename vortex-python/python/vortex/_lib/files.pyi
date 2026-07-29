#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from collections.abc import Sequence
from typing import final

import pyarrow as pa

from vortex.type_aliases import IntoProjection

from . import CosStore, HfStore
from .dtype import DType
from .expr import Expr
from .iter import ArrayIterator
from .store import ObjectStore

@final
class VortexFiles:
    @property
    def file_count(self) -> int: ...
    @property
    def dtype(self) -> DType: ...
    def schema(self) -> pa.Schema: ...
    def count_rows(self, *, expr: Expr | None = None) -> int: ...
    def scan(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        ordered: bool = True,
    ) -> ArrayIterator: ...
    def to_arrow(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        schema: pa.Schema | None = None,
        ordered: bool = True,
    ) -> pa.RecordBatchReader: ...

def open_files(
    paths: str | Sequence[str],
    *,
    store: ObjectStore | CosStore | HfStore | None = None,
) -> VortexFiles: ...
