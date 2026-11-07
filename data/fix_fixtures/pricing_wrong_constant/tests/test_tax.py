import unittest
from pricing.tax import vat_part, with_vat


class VatTests(unittest.TestCase):
    def test_gross(self):
        self.assertEqual(with_vat(1000), 1200)

    def test_vat_part(self):
        self.assertEqual(vat_part(250), 50)

    def test_zero(self):
        self.assertEqual(with_vat(0), 0)
