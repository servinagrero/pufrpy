import unittest

from pufrpy import *
import numpy as np


class TestUtils(unittest.TestCase):
    def test_rbits(self):
        v = rbits(10)
        self.assertEqual(v.shape, (10,))
        m = rbits((2, 5))
        self.assertEqual(m.shape, (2, 5))
        arr = rbits((3, 2, 5))
        self.assertEqual(arr.shape, (3, 2, 5))
        P = np.random.uniform(0, 1, 10)
        with self.assertRaises(AssertionError):
            rbits((2, 10), P)

    def test_entropy(self):
        v = rbits(100)
        ent = entropy_bits(v)
        self.assertLessEqual(1 - ent, 0.1)

    def test_entropy_prob(self):
        p = np.random.uniform(0, 1, 100)
        ent = entropy_prob(p)
        self.assertEqual(len(p[np.isnan(p)]), 0)

    def test_hamming_weight(self):
        v = np.random.choice([0, 1, np.nan], 100, replace=True)
        ones = len(v[v == 1])
        zeros = len(v[v == 0])
        self.assertEqual(hamming_weight(v, True), ones / (ones + zeros))

    def test_hamming_dist(self):
        v = np.random.choice([0, 1, np.nan], 100, replace=True)
        hd = hamming_dist(v, v, True)
        self.assertEqual(hamming_dist(v, v, True), 0)
