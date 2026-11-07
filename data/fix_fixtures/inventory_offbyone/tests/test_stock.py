import unittest
from inventory.stock import restock_needed, total_units


class RestockTests(unittest.TestCase):
    def test_at_threshold_needs_restock(self):
        self.assertEqual(restock_needed({"bolts": 5, "nuts": 9}, 5), ["bolts"])

    def test_below_threshold(self):
        self.assertEqual(restock_needed({"a": 1, "b": 2, "c": 10}, 3), ["a", "b"])

    def test_total(self):
        self.assertEqual(total_units({"a": 2, "b": 3}), 5)
