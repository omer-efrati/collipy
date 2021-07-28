import unittest
import numpy as np
from simulator.helpers import PhysicalQuantity


class TestQuant(unittest.TestCase):

    def test_negative_uncertainty(self):
        with self.assertRaises(ValueError):
            PhysicalQuantity('m1', 100, -5, 'kg')
        with self.assertRaises(ValueError):
            PhysicalQuantity('L', 100, 0, 'm')

    def test_validity_after_conversion(self):
        mass = PhysicalQuantity('m1', 100, 5, 'kg')
        mass.convert_units(10**3, 'g')
        self.assertTrue(mass.name == 'm1')
        self.assertTrue(mass.value == 100_000)
        self.assertTrue(mass.uncertainty == 5_000)
        self.assertTrue(mass.units == 'g')
        with self.assertRaises(ValueError):
            mass.convert_units(-10**-3, '-kg')

    def test_n_sigma(self):
        length1 = PhysicalQuantity('L1', 10, 1, 'cm')
        length2 = PhysicalQuantity('L2', 10, 1, 'cm')
        length3 = PhysicalQuantity('L3', 1, 1, 'm')
        self.assertTrue(length1.n_sigma(length2) == length2.n_sigma(length1) == 0)
        self.assertTrue(length1.n_sigma(5.5) == 4.5)
        self.assertTrue(length1.n_sigma(5) == 5)
        self.assertTrue(length3.n_sigma(0) == 1)
        self.assertTrue(length3.n_sigma(-5) == 6)
        with self.assertRaises(ValueError):
            length3.n_sigma(length1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
