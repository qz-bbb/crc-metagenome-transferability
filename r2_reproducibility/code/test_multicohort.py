import unittest
import numpy as np
import pandas as pd
import multicohort_core as m

class Boundaries(unittest.TestCase):
    def test_unequal_groups_constant_decimal_not_significant(self):
        y=np.array([0]*39+[1]*18); x=np.full((57,2),1.7)
        effect,p,q=m.welch_effects(x,y)
        np.testing.assert_array_equal(effect,[0,0]); np.testing.assert_array_equal(p,[1,1])
        np.testing.assert_array_equal(m.bootstrap_effects(x,y,100,42),np.zeros((100,2)))
    def test_roundoff_does_not_count_as_direction(self):
        self.assertEqual(m.direction_fraction(np.array([4e-16]),np.array([1.])),0.)
    def test_nondetected_taxon_cannot_biologically_replicate(self):
        self.assertTrue(hasattr(m,'bootstrap_directions'),'detection-aware endpoint missing')
        x=np.array([[0.,0.],[0.,0.],[2.,2.],[2.,2.]])
        raw=np.array([[1.,0.],[1.,0.],[2.,0.],[2.,0.]])
        boot=m.bootstrap_directions(x,np.array([0,0,1,1]),raw,np.array([1.,1.]),100,42)
        np.testing.assert_array_equal(boot,np.tile([True,False],(100,1)))
    def test_zero_effect_is_discordant_not_dropped(self):
        self.assertAlmostEqual(m.direction_fraction(np.array([2.,-1.,0.]),np.array([1.,-3.,5.])),2/3)
    def test_opposite_effect_does_not_replicate(self):
        self.assertEqual(m.direction_fraction(np.array([-2.,1.]),np.array([1.,-1.])),0.)
    def test_missing_or_nonfinite_effect_rejected(self):
        with self.assertRaises(ValueError): m.direction_fraction(np.array([np.nan]),np.array([1.]))
    def test_bootstrap_keeps_taxa_paired_and_is_deterministic(self):
        x=np.array([[0.,0.],[0.,0.],[2.,-2.],[2.,-2.]])
        a=m.bootstrap_effects(x,np.array([0,0,1,1]),100,42)
        np.testing.assert_array_equal(a,np.tile([2.,-2.],(100,1)))
        np.testing.assert_array_equal(a,m.bootstrap_effects(x,np.array([0,0,1,1]),100,42))
    def test_bootstrap_rejects_single_group(self):
        with self.assertRaises(ValueError): m.bootstrap_effects(np.ones((3,2)),np.ones(3),10,42)
    def test_welch_constant_equal_gets_p_one(self):
        effect,p,q=m.welch_effects(np.ones((6,2)),np.array([0,0,0,1,1,1]))
        np.testing.assert_array_equal(effect,[0,0]); np.testing.assert_array_equal(p,[1,1]); np.testing.assert_array_equal(q,[1,1])
    def test_id_gate_detects_training_overlap(self):
        meta=pd.DataFrame({'Sample Source ID':['x','y'],'Study condition':['CRC','Control']})
        with self.assertRaises(ValueError): m.check_ids(meta,['x','y'],{'x'})
    def test_id_gate_rejects_profile_only_id(self):
        meta=pd.DataFrame({'Sample Source ID':['x','y'],'Study condition':['CRC','Control']})
        with self.assertRaises(ValueError): m.check_ids(meta,['x','y','z'],set())
    def test_id_gate_rejects_duplicate_id(self):
        meta=pd.DataFrame({'Sample Source ID':['x','x'],'Study condition':['CRC','Control']})
        with self.assertRaises(ValueError): m.check_ids(meta,['x'],set())

if __name__=='__main__': unittest.main()

