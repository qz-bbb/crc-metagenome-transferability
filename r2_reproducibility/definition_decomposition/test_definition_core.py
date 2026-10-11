import unittest
import numpy as np

try:
    import definition_core as core
except ImportError:
    core = None


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, 'definition_core must implement the fixed four-effect contract')

    def test_zero_median_is_not_zero_mean(self):
        # Replacing mean by median, or treating two zero medians as concordant, fails.
        raw=np.array([[0.,10.],[0.,10.],[9.,1.],[0.,10.],[0.,10.],[3.,7.]])
        y=np.array([1,1,1,0,0,0])
        effects, detected=core.effect_grid(raw,y,0.1)
        np.testing.assert_allclose(effects['raw_mean'],[2.,-2.],atol=1e-14)
        np.testing.assert_array_equal(effects['raw_median'],[0.,0.])
        self.assertEqual(core.concordance(effects['raw_median'],[1,-1],detected).tolist(),[False,False])

    def test_clr_uses_all_frozen_coordinates(self):
        # Omitting centering or centering across subjects fails.
        z=core.clr(np.array([[1.,3.],[3.,1.]]),1.)
        np.testing.assert_allclose(z, [[-0.34657359027997264,0.34657359027997264],[0.34657359027997264,-0.34657359027997264]],atol=1e-14)
        np.testing.assert_allclose(z.sum(axis=1),0,atol=1e-14)

    def test_undetected_feature_never_concordant(self):
        self.assertEqual(core.concordance([2.,-2.,0.,1e-14],[1,-1,1,1],[False,True,True,True]).tolist(),[False,True,False,False])

    def test_agreement_denominator_keeps_zero_studies(self):
        # Dropping zeros produces 1 instead of 2/3.
        agreement=core.agreement(np.array([[True,False],[False,False],[True,True]]))
        np.testing.assert_allclose(agreement,[2/3,1/3])

    def test_both_classes_required(self):
        with self.assertRaises(ValueError):core.effect_grid(np.ones((3,2)),np.ones(3),.1)

    def test_invalid_raw_and_pseudocount_rejected(self):
        for raw,pseudo in [(np.array([[1.,-1.],[1.,1.]]),.1),(np.array([[1.,np.nan],[1.,1.]]),.1),(np.ones((2,2)),0.)]:
            with self.assertRaises(ValueError):core.effect_grid(raw,[1,0],pseudo)

    def test_shape_and_nonbinary_labels_rejected(self):
        for y in ([1,0,1],[1,2]):
            with self.assertRaises(ValueError):core.effect_grid(np.ones((2,2)),y,.1)

    def test_four_definitions_returned(self):
        effects,_=core.effect_grid(np.array([[1.,9.],[4.,6.],[2.,8.],[3.,7.]]),[1,1,0,0],.1)
        self.assertEqual(set(effects),{'raw_mean','raw_median','clr_mean','clr_median'})
        np.testing.assert_allclose(effects['raw_mean'],[0,0],atol=1e-14)

    def test_nonfinite_reference_sign_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                core.concordance([1.], [value], [True])

if __name__=='__main__':unittest.main(verbosity=2)

