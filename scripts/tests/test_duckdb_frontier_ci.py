# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Protect the independent result gate and balanced comparison order."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("frontier_ci", Path(__file__).parents[1] / "duckdb-frontier-ci.py")
CI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CI)


class ResultGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, values, dtype="BIGINT"):
        content = f'duckdb-result-v1 columns=1 rows={len(values)}\ncolumn=0 name="v" logical_type="{dtype}"\n'
        for i, value in enumerate(values):
            content += f"row={i}\nvalue=0 {value}\n"
        path = self.root / name
        path.write_text(content)
        return path

    def test_large_integer_changes_are_not_hidden_by_float_rounding(self):
        left = self.write("left", ['text="9007199254740992"'])
        right = self.write("right", ['text="9007199254740993"'])
        with self.assertRaisesRegex(ValueError, "Value mismatch"):
            CI.assert_results_equal(left, right)

    def test_float_aggregation_tolerance_is_tight(self):
        left = self.write("left", ['text="1.0"'], "DOUBLE")
        near = self.write("near", ['text="1.000000000001"'], "DOUBLE")
        far = self.write("far", ['text="1.001"'], "DOUBLE")
        CI.assert_results_equal(left, near)
        with self.assertRaisesRegex(ValueError, "Value mismatch"):
            CI.assert_results_equal(left, far)

    def test_multisets_keep_duplicates_and_null_distinct_from_empty(self):
        left = self.write("left", ['text=""', "null", 'text="x"'], "VARCHAR")
        right = self.write("right", ['text="x"', 'text=""', "null"], "VARCHAR")
        bad = self.write("bad", ['text="x"', "null", "null"], "VARCHAR")
        CI.assert_results_equal(left, right)
        with self.assertRaisesRegex(ValueError, "Value mismatch"):
            CI.assert_results_equal(left, bad)

    def test_rust_unicode_escape_and_truncated_result(self):
        path = self.write("escaped", [r'text="\u{1b}\u{1f600}"'], "VARCHAR")
        self.assertEqual(CI.result(path)[1], [["\x1b😀"]])
        path.write_text(path.read_text().replace("rows=1", "rows=2"))
        with self.assertRaisesRegex(ValueError, "Truncated"):
            CI.result(path)

    def test_schema_is_checked(self):
        left = self.write("left", ['text="1"'])
        right = self.write("right", ['text="1"'], "VARCHAR")
        with self.assertRaisesRegex(ValueError, "Schema"):
            CI.assert_results_equal(left, right)

    def test_all_six_orders_are_unique_and_position_balanced(self):
        orders = [CI.arm_order(i) for i in range(6)]
        self.assertEqual(len(set(orders)), 6)
        for position in range(3):
            for arm in CI.ARMS:
                self.assertEqual(sum(order[position] == arm for order in orders), 2)


if __name__ == "__main__":
    unittest.main()
