import numpy as np
from typing import Union
import unittest

class EWMA:

    def __init__(self, halflife: float, initial_value: Union[float, None]=None):
        self._x = initial_value
        self._gamma = 2.0**(-1.0 / halflife)

    def update(self, x):
        if self._x == None:
            self._x = x
        else:
            self._x *= self._gamma
            self._x += (1.0 - self._gamma) * x

    def get(self):
        if self._x is not None:
            return self._x
        else:
            raise ValueError("Called get on EWMA without any insertion and no initial value.")

class TestEWMA(unittest.TestCase):

    def test_initial_value(self):
        ewma = EWMA(halflife=10.0, initial_value=5.0)
        self.assertEqual(ewma.get(), 5.0)

    def test_no_initial_value(self):
        ewma = EWMA(halflife=10.0)
        self.assertRaises(ValueError)

    def test_static(self):
        ewma = EWMA(halflife=10.0)
        ewma.update(10.0)
        self.assertEqual(ewma.get(), 10.0)
        ewma.update(10.0)
        self.assertEqual(ewma.get(), 10.0)
        ewma.update(10.0)
        self.assertEqual(ewma.get(), 10.0)
        ewma.update(10.0)
        self.assertEqual(ewma.get(), 10.0)

    def test_decay(self):
        ewma = EWMA(halflife=10.0)
        ewma.update(10.0)
        self.assertEqual(ewma.get(), 10.0)
        
        for _ in range(10):
            ewma.update(0.0)
        self.assertEqual(ewma.get(), 10.0/2.0)
        
        for _ in range(10):
            ewma.update(0.0)
        self.assertEqual(ewma.get(), 10.0/4.0)


    def test_decay_nonzero(self):
        ewma = EWMA(halflife=10.0)
        ewma.update(10.0)
        self.assertEqual(ewma.get(), 10.0)
        
        for _ in range(10):
            ewma.update(20.0)
        np.testing.assert_almost_equal(ewma.get(), 20.0 + (10.0 - 20.0)/2.0, decimal=14)
        
        for _ in range(10):
            ewma.update(20.0)
        np.testing.assert_almost_equal(ewma.get(), 20.0 + (10.0 - 20.0)/4.0, decimal=14)

