import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("./frbfuncs/"))
import ktau as kt

from sympy.utilities.iterables import multiset_permutations

class TestKTau(unittest.TestCase):
    def test_ktau_fast(self):
        y1 = np.array([1,2,3,4,5])
        x1 = np.array([0,0.1,0.2,0.3,0.4]) # essentially allow all points
        
        #essentially test all 5! permutations
        for p in multiset_permutations(y1):
            p = np.array(p)
            self.assertAlmostEqual(kt.ktau_fast(p, x1, x1), kt.ktau_E(p, x1, x1), delta=1e-10)
            self.assertAlmostEqual(kt.ktau_fast(p, x1, x1, k=2), kt.ktau_E(p, x1, x1, k=2), delta=1e-10)
        
        y2 = np.array([1,2,3,4,5])
        x2 = np.array([0, 1, 1, 2, 2]) #test for differences in same ylim, cutting off points
        
        for p in multiset_permutations(y2):
            p = np.array(p)
            self.assertAlmostEqual(kt.ktau_fast(p, x2, x2), kt.ktau_E(p, x2, x2), delta=1e-10)
            self.assertAlmostEqual(kt.ktau_fast(p, x2, x2, k=2), kt.ktau_E(p, x2, x2, k=2), delta=1e-10)
            
        y3 = np.array([1,2,2,3,3]) # a bunch of ties
        x3 = np.array([0,0,1,1,2])
        
        for p in multiset_permutations(y3):
            p = np.array(p)
            self.assertAlmostEqual(kt.ktau_fast(p, x3, x3), kt.ktau_E(p, x3, x3), delta=1e-10)
            self.assertAlmostEqual(kt.ktau_fast(p, x3, x3, k=2), kt.ktau_E(p, x3, x3, k=2), delta=1e-10)
        
        seed = 0
        rng = np.random.default_rng(seed)
        y4 = rng.random(10000)
        x4 = rng.random(10000)
        k = rng.random()*5 #random number from 0-5
        
        self.assertAlmostEqual(kt.ktau_fast(y4, x4, x4, k=k), kt.ktau_E(y4, x4, x4, k=k), delta=1e-10)
        
        
unittest.main(argv=[''], verbosity=2, exit=False)