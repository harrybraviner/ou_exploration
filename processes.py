import numpy as np
from typing import Union
import unittest
from averages import EWMA, DelayedStat

class OUProcess:

    def __init__(self, decay_theta, sigma, initial_value: np.ndarray, random_seed: int=1234):

        self._rng = np.random.RandomState(random_seed)
        self._decay_theta = decay_theta
        self._sigma = sigma

        self._initial_value = initial_value.copy()
        self._state = self._initial_value.copy()

        self._dim = self._initial_value.shape

        self.reset()

    def reset(self):
        self._state = self._initial_value.copy()

    def sample(self):
        noise = self._rng.normal(loc=0.0, scale=1.0, size=self._dim) * self._sigma

        self._state *= (1.0 - self._decay_theta)
        self._state += noise

        return self._state.copy()


class OUProcessWithMemory:

    def __init__(self, decay_theta, sigma, initial_value: np.ndarray, av_halflife: int=30, av_delay: int=10, av_sign: float=1.0, random_seed: int=1234):

        self._rng = np.random.RandomState(random_seed)
        self._decay_theta = decay_theta
        self._sigma = sigma
        self._av_halflife = av_halflife
        self._av_delay = av_delay
        self._av_sign = av_sign

        self._initial_value = initial_value.copy()
        self._state = self._initial_value.copy()

        self._dim = self._initial_value.shape

        self.reset()

        self._ewmas = [DelayedStat(EWMA(halflife=self._av_halflife, initial_value=0.0), delay=self._av_delay) for _ in self._dim]

    def reset(self):
        self._state = self._initial_value.copy()
        self._ewmas = [DelayedStat(EWMA(halflife=self._av_halflife, initial_value=0.0), delay=self._av_delay) for _ in self._dim]

    def sample(self):
        noise = self._rng.normal(loc=0.0, scale=1.0, size=self._dim) * self._sigma

        av_state = np.array([av.get() for av in self._ewmas], dtype=np.float32).reshape(self._dim)
        for av, s in zip(self._ewmas, self._state):
            av.update(s)

        self._state *= (1.0 - self._decay_theta)
        self._state += (self._decay_theta) * self._av_sign * av_state
        self._state += noise

        return self._state.copy()



class TestOUProcess(unittest.TestCase):

    def test_runs_without_crash(self):

        p = OUProcess(decay_theta=0.1, sigma=0.01, initial_value=np.zeros(dtype=np.float32, shape=(1,)))

        for _ in range(10):
            p.sample()

class TestOUProcessWithMemory(unittest.TestCase):

    def test_runs_without_crash(self):

        p = OUProcessWithMemory(decay_theta=0.1, sigma=0.01, initial_value=np.zeros(dtype=np.float32, shape=(1,)))

        for _ in range(10):
            p.sample()

