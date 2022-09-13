import unittest
import numpy as np
import sys
#import os
from scipy.special import erf
sys.path.append("./frbfuncs")
import cosmology as c
import deprecated_cosmology as d

# test cosmology.py
class TestCosmology(unittest.TestCase):
    
    def test_set_default(self):
        # change COSM_MODEL
        c.COSM_MODEL = [1,2,5,3,6]
        c.H_0 = c.O_M = c.O_L = c.O_B = 69
        
        # verify that change has taken place
        self.assertEqual(c.COSM_MODEL, [1,2,5,3,6])
        self.assertEqual(c.O_M, 69)
        self.assertEqual(c.H_0, 69)
        
        # set to default
        c.set_default()
        
        # verify all cosmological parameters are set to default
        self.assertEqual(c.COSM_MODEL, [0.286, 0.714, 0.049, 70])
        self.assertEqual(c.O_M, 0.286)
        self.assertEqual(c.O_L, 0.714)
        self.assertEqual(c.O_B, 0.049)
        self.assertEqual(c.H_0, 70*c.KMSMPC_TO_S)

    def test_update_cmodel(self):
        # update COSM_MODEL
        new_model = [0.284, 0.755, 0.010, 50]
        c.update_cmodel(new_model)
        
        # check that all aspects of the cosmological model are modified, as expected
        self.assertEqual(c.COSM_MODEL, [0.284, 0.755, 0.010, 50])
        self.assertEqual(c.H_0, 50*c.KMSMPC_TO_S)
        self.assertEqual(c.O_M, 0.284)
        self.assertEqual(c.O_L, 0.755)
        self.assertEqual(c.O_B, 0.010)
        
        # check to see that COSM_MODEL updates independently
        new_model.append(69)
        self.assertEqual(c.COSM_MODEL, [0.284, 0.755, 0.010, 50])
        
        # set_default() already checked by above function; this is simply to reset parameters for future use
        c.set_default()
    
    def test_sort_by_first(self):
        # test case 1
        l1, l2 = c.sort_by_first([4,3,5,-1,2], [5,69,8,72,-11])
        self.assertEqual(l1.tolist(), [-1,2,3,4,5])
        self.assertEqual(l2.tolist(), [72,-11,69,5,8])
        
        # test case 2
        l1, l2 = c.sort_by_first([5,4,3,2], [-5,-6,-7,8])
        self.assertEqual(l1.tolist(), [2,3,4,5])
        self.assertEqual(l2.tolist(), [8,-7,-6,-5])
        
        # test case 3 (check if sorts in place)
        l1, l2 = c.sort_by_first([9,9,3,9,9], [5,-2,3,-4,99])
        self.assertEqual(l1.tolist(), [3,9,9,9,9])
        self.assertEqual(l2.tolist(), [3,5,-2,-4,99])
        
        # test case 4 (test case 1 but with np array)
        l1, l2 = c.sort_by_first(np.array([4,3,5,-1,2]), np.array([5,69,8,72,-11]))
        self.assertEqual(l1.tolist(), [-1,2,3,4,5])
        self.assertEqual(l2.tolist(), [72,-11,69,5,8])
        
        # test case 5 (test case 2 but with np array)
        l1, l2 = c.sort_by_first(np.array([5,4,3,2]), np.array([-5,-6,-7,8]))
        self.assertEqual(l1.tolist(), [2,3,4,5])
        self.assertEqual(l2.tolist(), [8,-7,-6,-5])
        
        # test case 6 (test case 3 but with np array)
        l1, l2 = c.sort_by_first(np.array([9,9,3,9,9]), np.array([5,-2,3,-4,99]))
        self.assertEqual(l1.tolist(), [3,9,9,9,9])
        self.assertEqual(l2.tolist(), [3,5,-2,-4,99])
    
    def test_integrate_mult_ends(self):
        # initialize data and known functions, integrated
        cubed = lambda x: x**3
        int_cubed = lambda x: x**4/4
        expsq = lambda x: np.exp(-x**2)
        int_expsq = lambda x: (np.pi)**0.5/2*erf(x)
        start = 2.3
        end1 = 7.4
        end2 = np.array([7.4, 3.74, 8.123, 3, 6.68, 99])
        
        ### check polynomial integrates correctly ###
        # single value
        self.assertAlmostEqual(c.integrate_mult_ends(cubed, start, end1), int_cubed(end1)-int_cubed(start), delta=1e-8)
        # np array of end values
        np.testing.assert_almost_equal(c.integrate_mult_ends(cubed, start, end2), int_cubed(end2)-int_cubed(start), 8)
        
        ### check integral with no elementary form integrates correctly ###
        # single value
        self.assertAlmostEqual(c.integrate_mult_ends(expsq, start, end1), int_expsq(end1)-int_expsq(start), delta=1e-8)
        # np array of end values
        np.testing.assert_almost_equal(c.integrate_mult_ends(expsq, start, end2), int_expsq(end2)-int_expsq(start), 8)
        
    def test_E(self):
        # test single values
        self.assertEqual(c.E(0), (c.O_M + c.O_L)**0.5)
        self.assertEqual(c.E(1), (8.0*c.O_M + c.O_L)**0.5)
        self.assertEqual(c.E(0.5), (3.375*c.O_M + c.O_L)**0.5)
        self.assertEqual(c.E(55.19482), ((56.19482)**3*c.O_M + c.O_L)**0.5)
        
        # test np array of values
        ans = np.array([(c.O_M + c.O_L)**0.5, (8.0*c.O_M + c.O_L)**0.5, (3.375*c.O_M + c.O_L)**0.5, ((56.19482)**3*c.O_M + c.O_L)**0.5])
        self.assertEqual(c.E(np.array([0,1,0.5,55.19482])).tolist(), ans.tolist())
    
    def test_D_C(self):
        # test single values
        self.assertAlmostEqual(c.D_C(2.12384), d.D_C_old(2.12384), delta=1e-7)
        self.assertAlmostEqual(c.D_C(2.12384), 1.26580797970764, delta=1e-11) # value calcualted through integral-calculator
        self.assertAlmostEqual(c.D_C(55.123854), 2.870559850206997, delta=1e-11) # value calcualted through integral-calculator
        
        # test np array of values
        input = np.array([50,1,3,25,4])
        np.testing.assert_almost_equal(c.D_C(input), d.D_C_old(input), 7)
        np.testing.assert_almost_equal(c.D_C(input), np.array([2.846083449712865, 0.7779423776121797, 1.504993571457544, 2.636332485477817, 1.699640679205983]), 11)

    def test_D_L(self):
        # test single values
        self.assertAlmostEqual(c.D_L(2.12384), 3.12384*1.26580797970764, delta=1e-11)
        self.assertAlmostEqual(c.D_L(55.123854), 56.123854*2.870559850206997, delta=1e-11) # value calcualted through integral-calculator
        
        # test np array of values
        exp = np.array([2.846083449712865, 0.7779423776121797, 1.504993571457544, 2.636332485477817, 1.699640679205983]) * np.array([51,2,4,26,5])
        np.testing.assert_almost_equal(c.D_L(np.array([50,1,3,25,4])), exp, 11)
    
    def test_dDMdz_Arcus(self):
        # test a variety of single values
        self.assertAlmostEqual(c.dDMdz_Arcus(0), 1010.44805898, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Arcus(1.5), 1109.61896919, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Arcus(2.293), 1006.60923609, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Arcus(4.87), 664.356689675, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Arcus(6.11991), 604.853405924, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Arcus(9.326), 0, delta=1e-8)
        
        #test np array of values
        input = np.array([6.11991, 4.87, 0, 1.5, 9.326, 2.293])
        exp = np.array([604.853405924, 664.356689675, 1010.44805898, 1109.61896919, 0, 1006.60923609])
        np.testing.assert_almost_equal(c.dDMdz_Arcus(input), exp, 7)
    
    def test_dDMdz_Zhang(self):
        # test a variety of single values
        self.assertAlmostEqual(c.dDMdz_Zhang(0), 848.776369544, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Zhang(1.5), 932.07993412, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Zhang(2.293), 845.551758312, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Zhang(4), 702.798992847, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Zhang(6.11991), 592.756337805, delta=1e-8)
        self.assertAlmostEqual(c.dDMdz_Zhang(9.326), 493.346886956, delta=1e-8)
        
        # test np array of values
        input = np.array([6.11991, 4, 0, 1.5, 9.326, 2.293])
        exp = np.array([592.756337805, 702.798992847, 848.776369544, 932.07993412, 493.346886956, 845.551758312])
        np.testing.assert_almost_equal(c.dDMdz_Zhang(input), exp, 7)
    
    def test_DM(self):
        ### test single values ###
        # default should be Zhang
        self.assertAlmostEqual(c.DM(1.75432), d.DM_V1(1.75432, dDMdz=c.dDMdz_Zhang), delta=1)
        # Arcus
        self.assertAlmostEqual(c.DM(1.75432, dDMdz=c.dDMdz_Arcus), d.DM_V1(1.75432, dDMdz=c.dDMdz_Arcus), delta=1)
        # Zhang
        self.assertAlmostEqual(c.DM(1.75432, dDMdz=c.dDMdz_Zhang), d.DM_V1(1.75432, dDMdz=c.dDMdz_Zhang), delta=1)
        # custom function
        
        ### test np array of values ###
        input = np.array([9, 1.7, 3.1257, 0.135, 0.743, 5.812, 7.952785])
        # default should be Zhang
        np.testing.assert_almost_equal(c.DM(input), d.DM_V1(input, dDMdz=c.dDMdz_Zhang), 0)
        # Arcus (DM(z) should still work for z > 8, just constant)
        np.testing.assert_almost_equal(c.DM(input, dDMdz=c.dDMdz_Arcus), d.DM_V1(input, dDMdz=c.dDMdz_Arcus), 0)
        # Zhang
        np.testing.assert_almost_equal(c.DM(input, dDMdz=c.dDMdz_Zhang), d.DM_V1(input, dDMdz=c.dDMdz_Zhang), 0)
    
    def test_set_fast_DM(self):
        ### test default settings are as expected
        z1 = np.linspace(0, 8, 2000)
        np.testing.assert_almost_equal(z1, c._z, 7)
        np.testing.assert_almost_equal(c.DM(z1, dDMdz=c.dDMdz_Arcus), c._DM_Arcus, 7)
        np.testing.assert_almost_equal(c.DM(z1, dDMdz=c.dDMdz_Zhang), c._DM_Zhang, 7)
        
        ### test setting it to a different setting
        c.set_fast_DM(3, 5, 876)
        z2 = np.linspace(3, 5, 876)
        np.testing.assert_almost_equal(z2, c._z)
        np.testing.assert_almost_equal(c.DM(z2, dDMdz=c.dDMdz_Arcus), c._DM_Arcus, 7)
        np.testing.assert_almost_equal(c.DM(z2, dDMdz=c.dDMdz_Zhang), c._DM_Zhang, 7)
        
        ### test setting it back to default settings
        c.set_fast_DM()
        np.testing.assert_almost_equal(z1, c._z, 7)
        np.testing.assert_almost_equal(c.DM(z1, dDMdz=c.dDMdz_Arcus), c._DM_Arcus, 7)
        np.testing.assert_almost_equal(c.DM(z1, dDMdz=c.dDMdz_Zhang), c._DM_Zhang, 7)
        pass
    
    def test_z_DM(self):
        ### test single value of DM ###
        DM1 = 719.27384
        # default should work
        self.assertAlmostEqual(c.DM(c.z_DM(DM1)), DM1, delta = 1e-5)
        # Arcus
        self.assertAlmostEqual(c.DM(c.z_DM(DM1, "Arcus"), dDMdz=c.dDMdz_Arcus), DM1, delta = 1e-5)
        # Zhang
        self.assertAlmostEqual(c.DM(c.z_DM(DM1, "Zhang"), dDMdz=c.dDMdz_Zhang), DM1, delta = 1e-5)
        
        ### test single value of z ###
        z1 = 0.517394
        # default should work
        self.assertAlmostEqual(c.z_DM(c.DM(z1)), z1, delta = 1e-9)
        # Arcus
        self.assertAlmostEqual(c.z_DM(c.DM(z1, dDMdz=c.dDMdz_Arcus), "Arcus"), z1, delta = 1e-8)
        # Zhang
        self.assertAlmostEqual(c.z_DM(c.DM(z1, dDMdz=c.dDMdz_Zhang), "Zhang"), z1, delta = 1e-9)
        
        ### test multiple values of DM ###
        DM2 = np.array([719.27384, 318.9591, 951.7174, 1502.58448, 6012.2384, 4815.234])
        # default should work
        np.testing.assert_almost_equal(c.DM(c.z_DM(DM2)), DM2, 5)
        # Arcus
        np.testing.assert_almost_equal(c.DM(c.z_DM(DM2, func="Arcus"), dDMdz=c.dDMdz_Arcus), DM2, 3) # less accurate + slower than Zhang.
        # Zhang
        np.testing.assert_almost_equal(c.DM(c.z_DM(DM2, func="Zhang"), dDMdz=c.dDMdz_Zhang), DM2, 5)
        
        ### test multiple values of z ###
        z2 = np.array([0.0001534, 0.517394, 1.1737, 3.9999, 2.48173, 5.83271, 7.83021])
        # default should work
        np.testing.assert_almost_equal(c.z_DM(c.DM(z2)), z2, 8)
        # Arcus
        np.testing.assert_almost_equal(c.z_DM(c.DM(z2, dDMdz=c.dDMdz_Arcus), "Arcus"), z2, 5) # a lot less accurate + slower than Zhang
        # Zhang
        np.testing.assert_almost_equal(c.z_DM(c.DM(z2, dDMdz=c.dDMdz_Zhang), "Zhang"), z2, 8)
    
    def test_z_DM_fast_mode(self):
        #TODO
        pass
    
    def test_E_v(self):
        #TODO
        pass

    def test_reset_used(self):
        #TODO
        pass

    def test_z_E(self):
        #TODO
        pass

unittest.main(argv=[''], verbosity=2, exit=False)