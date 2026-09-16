# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Reject incomplete or invalid benchmark result matrices; no GPU required."""

import itertools
import unittest
from unittest.mock import patch

import reproduce

QUERIES = (1, 5, 6, 9, 10)


def result_data(query: int) -> dict:
    states = []
    engines = ("binaryop", "ast", "transform") if query == 9 else (None,)
    for fmt, workload, cache, engine in itertools.product(
        ("parquet", "vortex"),
        ("read", f"q{query}"),
        ("warm", "cold"),
        engines,
    ):
        axes = dict(scale_factor="1", format=fmt, workload=workload, cache=cache)
        if engine is not None:
            axes["engine"] = engine
        states.append(
            {
                "name": str(axes),
                "device": 0,
                "axis_values": [{"name": k, "value": v} for k, v in axes.items()],
                "summaries": [
                    {
                        "tag": "nv/cold/time/cpu/mean",
                        "data": [{"name": "value", "value": 0.25}],
                    }
                ],
            }
        )
    return {"benchmarks": [{"name": f"ndsh_q{query}_local", "states": states}]}


class ResultValidationTests(unittest.TestCase):
    def test_complete_matrices(self):
        for query in QUERIES:
            with self.subTest(query=query):
                data = result_data(query)
                data["benchmarks"][0]["states"].reverse()
                reproduce.validate_results(data, query, 1.0)

    def test_rejects_incomplete_skipped_and_untimed_matrices(self):
        for query in (1, 9):
            data = result_data(query)
            benchmark = data["benchmarks"][0]
            states = benchmark["states"]
            state = states[0]
            summary = state["summaries"][0]
            scale_axis = next(axis for axis in state["axis_values"] if axis["name"] == "scale_factor")
            defects: list[tuple[str | int, dict, dict]] = [
                ("missing", benchmark, {"states": states[:-1]}),
                ("duplicate", benchmark, {"states": [*states[:-1], state]}),
                ("skipped", state, {"is_skipped": True}),
                ("untimed", state, {"summaries": []}),
                ("wrong_timer", summary, {"tag": "nv/cold/time/gpu/mean"}),
                ("wrong_scale", scale_axis, {"value": "10"}),
                ("wrong_name", benchmark, {"name": f"ndsh_q{query}_other"}),
                ("wrong_device", state, {"device": 1}),
            ]
            for name in ("format", "workload", "cache", "engine"):
                axes = [axis for axis in state["axis_values"] if axis["name"] != name]
                defects.append((name, state, {"axis_values": [*axes, {"name": name, "value": "unexpected"}]}))
            for value in (0, -1, "nan", "inf"):
                defects.append((value, summary["data"][0], {"value": value}))
            for defect, target, changes in defects:
                with self.subTest(query=query, defect=defect), patch.dict(target, changes):
                    with self.assertRaises(RuntimeError):
                        reproduce.validate_results(data, query, 1.0)


if __name__ == "__main__":
    unittest.main()
