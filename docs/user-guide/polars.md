# Polars

:::{warning}
The Polars integration is experimental. Polars' expression API is unstable and not all pushdown
expressions are currently supported.

If you run into any issues or are looking for more features related to this integration, please [file an issue](https://github.com/vortex-data/vortex/issues).
:::

Vortex integrates with Polars via {meth}`.VortexFile.to_polars`, which returns a
{class}`polars.LazyFrame` with column pruning and predicate pushdown.

```{doctest} pycon
>>> import vortex as vx
>>> import pyarrow.parquet as pq
>>>
>>> vx.io.write(pq.read_table("_static/example.parquet"), 'example.vortex')
>>>
>>> lf = vx.open('example.vortex').to_polars()
>>> lf = lf.select('tip_amount', 'fare_amount')
>>> lf = lf.head(3)
>>> lf.collect()
shape: (3, 2)
┌────────────┬─────────────┐
│ tip_amount ┆ fare_amount │
│ ---        ┆ ---         │
│ f64        ┆ f64         │
╞════════════╪═════════════╡
│ 0.0        ┆ 61.8        │
│ 5.1        ┆ 20.5        │
│ 16.54      ┆ 70.0        │
└────────────┴─────────────┘
```

## Scanning many files

Where `polars.scan_parquet` accepts a directory or a glob pattern, use {func}`vortex.open_files`
to gather many Vortex files into one {class}`.VortexFiles` and call
{meth}`.VortexFiles.to_polars` on it. All files must share the same dtype, and the resulting
{class}`polars.LazyFrame` prunes columns and pushes down predicates just as it does for a single
file.

```python
import polars as pl
import vortex as vx

# A directory, scanned recursively for *.vortex files.
lf = vx.open_files("nyc_taxi/").to_polars()

# Or an explicit glob pattern, or a list of paths.
lf = vx.open_files("s3://bucket/nyc_taxi/year=2024/*.vortex").to_polars()
lf = vx.open_files(["jan.vortex", "feb.vortex"]).to_polars()

lf.filter(pl.col("passenger_count") > 2).select("fare_amount").collect()
```

Rows are returned in file order, sorted by path. Pass `to_polars(ordered=False)` to read files
concurrently instead, which is faster but leaves the row order unspecified.

Alternatively, {meth}`.VortexFiles.to_dataset` exposes the same files through the
{class}`pyarrow.dataset.Dataset` interface, which Polars consumes via
`polars.scan_pyarrow_dataset`. Prefer `to_polars()` for Polars specifically - it pushes
richer predicates - and `to_dataset()` when the same object also needs to feed DuckDB, pandas, or
another Arrow dataset consumer.

## Unsupported predicates

Only some Polars expressions can be translated into Vortex expressions and pushed into the scan.
Anything Vortex cannot represent - string operations and arithmetic, for example - is instead
evaluated by the integration after reading, so results stay correct at the cost of reading more
data than a fully pushed-down filter would.
