import numpy as np
from typing import Union
from collections import deque
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

class DelayedStat:

    def __init__(self, underlying, delay: int):
        self._underlying = underlying
        self._delay = delay
        self._buffer = deque(maxlen=self._delay)

    def update(self, x):
        if len(self._buffer) == self._delay:
            self._underlying.update(self._buffer.popleft())
        self._buffer.append(x)

    def get(self):
        return self._underlying.get()

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

class TestDelayedStat(unittest.TestCase):

    def test_ewma_delay(self):
        ewma = EWMA(halflife=1.0)
        delayed_ewma = DelayedStat(underlying=ewma, delay=2)

        self.assertRaises(ValueError)
        delayed_ewma.update(1.0)
        self.assertRaises(ValueError)
        delayed_ewma.update(2.0)
        # Two updates, still in the buffer
        self.assertRaises(ValueError)

        delayed_ewma.update(2.0)
        # Value 1.0 has now entered EWMA
        np.testing.assert_almost_equal(delayed_ewma.get(), 1.0, decimal=14)

        delayed_ewma.update(2.0)
        np.testing.assert_almost_equal(delayed_ewma.get(), (1.0 + 2.0)/2.0, decimal=14)

    def test_ewma_delay_with_init(self):
        ewma = EWMA(halflife=1.0, initial_value=0.0)
        delayed_ewma = DelayedStat(underlying=ewma, delay=2)

        self.assertEqual(delayed_ewma.get(), 0.0)
        delayed_ewma.update(1.0)
        self.assertEqual(delayed_ewma.get(), 0.0)
        delayed_ewma.update(2.0)
        # Two updates, still in the buffer
        self.assertEqual(delayed_ewma.get(), 0.0)

        delayed_ewma.update(10.0)
        # Value 1.0 has now entered EWMA
        np.testing.assert_almost_equal(delayed_ewma.get(), 1.0/2.0, decimal=14)

        # Value 2.0 has now entered EWMA
        delayed_ewma.update(10.0)
        np.testing.assert_almost_equal(delayed_ewma.get(), 1.0/2.0 + (2.0 - 1.0/2.0)/2.0, decimal=14)

