import unittest

from torch import nn


class BrevitasModelEqualsTest(unittest.TestCase):
    def test_two_empty_models_are_unequal(self):
        self.assertNotEqual(nn.Sequential(), nn.Sequential())


if __name__ == "__main__":
    unittest.main()
