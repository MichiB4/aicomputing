"""Regression coverage for seeded, nested task sampling."""
from __future__ import annotations

import unittest

from evaluation.benchmarks.selection import SELECTION_ALGORITHM_VERSION, select_items


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.pool = [f"id{index:03d}" for index in range(1, 201)]

    def test_selection_is_deterministic_and_ignores_candidate_order_and_duplicates(self):
        expected = select_items(self.pool, 100, 42, salt="humaneval_plus")
        self.assertEqual(select_items(self.pool, 100, 42, salt="humaneval_plus"), expected)
        self.assertEqual(select_items(list(reversed(self.pool)), 100, 42,
                                      salt="humaneval_plus"), expected)
        self.assertEqual(select_items(self.pool[::2] + self.pool[1::2] + self.pool[:30],
                                      100, 42, salt="humaneval_plus"), expected)

    def test_every_sample_size_is_a_unique_prefix_even_when_selecting_the_full_pool(self):
        full = select_items(self.pool, len(self.pool), 42, salt="humaneval_plus")
        self.assertCountEqual(full, self.pool)
        self.assertNotEqual(full, sorted(self.pool))
        for size in (0, 1, 5, 10, 100, 199, 200, 201):
            with self.subTest(size=size):
                selected = select_items(self.pool, size, 42, salt="humaneval_plus")
                self.assertEqual(selected, full[:size])
                self.assertEqual(len(selected), min(size, len(self.pool)))
                self.assertEqual(len(set(selected)), len(selected))

    def test_empty_and_duplicate_only_pools(self):
        for size in (0, 1, 100):
            with self.subTest(size=size):
                self.assertEqual(select_items([], size, 42), [])
        self.assertEqual(select_items(["only"] * 10, 100, 42), ["only"])
        self.assertEqual(select_items(["only"] * 10, 0, 42), [])

    def test_invalid_sample_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            select_items(self.pool, -1, 42)
        for size in (True, False, 1.0, "10", None):
            with self.subTest(size=size), self.assertRaises(TypeError):
                select_items(self.pool, size, 42)

    def test_seed_and_benchmark_salt_change_the_selection(self):
        reference = select_items(self.pool, 10, 42, salt="humaneval_plus")
        self.assertNotEqual(select_items(self.pool, 10, 43, salt="humaneval_plus"), reference)
        selections = [select_items(self.pool, 10, 42, salt=benchmark)
                      for benchmark in ("humaneval_plus", "mbpp_plus", "gsm8k")]
        self.assertEqual(len({tuple(items) for items in selections}), 3)

    def test_smoke_sample_can_include_lexicographically_late_ids(self):
        smoke = select_items(self.pool, 10, 42, salt="humaneval_plus")
        selected_hundred = select_items(self.pool, 100, 42, salt="humaneval_plus")
        self.assertTrue(any(item > "id100" for item in smoke), smoke)
        # Sorting a sampled subset would bias small samples toward early IDs.
        self.assertNotEqual(smoke, sorted(selected_hundred)[:10])
        self.assertEqual(smoke, selected_hundred[:10])

    def test_selection_does_not_mutate_the_candidate_list(self):
        candidates = self.pool[::-1] + self.pool[:5]
        original = candidates.copy()
        select_items(candidates, 10, 42, salt="gsm8k")
        self.assertEqual(candidates, original)

    def test_selection_has_a_stable_recorded_order(self):
        self.assertEqual(SELECTION_ALGORITHM_VERSION, 6)
        pool = [f"task{index:02d}" for index in range(1, 13)]
        self.assertEqual(select_items(pool, 12, 42, salt="humaneval_plus"), [
            "task10", "task06", "task04", "task02", "task08", "task07",
            "task05", "task12", "task09", "task11", "task01", "task03",
        ])


if __name__ == "__main__":
    unittest.main()
