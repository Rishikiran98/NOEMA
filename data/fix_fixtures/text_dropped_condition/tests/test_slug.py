import unittest
from textkit.slug import slugify


class SlugTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_truncation_never_ends_with_dash(self):
        self.assertEqual(slugify("alpha beta gamma", max_length=11), "alpha-beta")

    def test_empty(self):
        self.assertEqual(slugify("!!!"), "")
