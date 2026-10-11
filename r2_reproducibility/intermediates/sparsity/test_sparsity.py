import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sparsity_generator import align_profile, describe, validate_output


class SparsityTests(unittest.TestCase):
    def test_integrity_gate_survives_optimized_python(self):
        result=subprocess.run([sys.executable,'-B','-O','-c',
            "from sparsity_generator import require_equal; require_equal(False,True,'mismatch')"],
            cwd=Path(__file__).resolve().parent,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('ValueError: mismatch',result.stderr)

    def profile(self):
        return pd.DataFrame({'a':[90.,10.], 'b':[80.,20.]},
                            index=['k__B|s__one','k__B|s__two'])

    def test_same_pipeline_nondetection_and_order(self):
        raw,present=align_profile(self.profile(),['b','a'],['k__B|s__two','k__B|s__absent'])
        np.testing.assert_array_equal(raw,[[20,0],[10,0]])
        self.assertEqual(present,[True,False])

    def test_missing_sample_rejected(self):
        with self.assertRaisesRegex(ValueError,'sample'):
            align_profile(self.profile(),['missing'],['k__B|s__one'])

    def test_duplicate_source_axis_rejected(self):
        p=self.profile();p.index=['k__B|s__one']*2
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            align_profile(p,['a'],['k__B|s__one'])

    def test_duplicate_requested_features_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            align_profile(self.profile(),['a'],['k__B|s__one']*2)

    def test_nonfinite_rejected(self):
        p=self.profile();p.iloc[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            align_profile(p,['a'],['k__B|s__one'])

    def test_negative_rejected(self):
        p=self.profile();p.iloc[0,0]=-1
        with self.assertRaisesRegex(ValueError,'non-negative'):
            align_profile(p,['a'],['k__B|s__one'])

    def test_wrong_unit_rejected(self):
        with self.assertRaisesRegex(ValueError,'percent'):
            align_profile(self.profile()/100,['a'],['k__B|s__one'])

    def test_zero_median_can_have_clr_concordance(self):
        raw=np.array([[0,100],[0,100],[5,95],[0,100],[0,100],[0,100]],float)
        a=describe(raw,np.array([1,1,1,0,0,0]),['a','b'],np.array([1,-1]),.01)
        self.assertTrue(a.loc[0,'zero_raw_median_difference'])
        self.assertFalse(a.loc[0,'raw_concordant'])
        self.assertTrue(a.loc[0,'clr_concordant'])

    def test_no_detection_never_clr_concordant(self):
        a=describe(np.array([[0,100],[0,90]],float),np.array([1,0]),['a','b'],np.array([-1,1]),.01)
        self.assertFalse(a.loc[0,'externally_detected'])
        self.assertFalse(a.loc[0,'clr_concordant'])

    def test_single_class_rejected(self):
        with self.assertRaisesRegex(ValueError,'Both classes'):
            describe(np.ones((2,2)),np.ones(2),['a','b'],np.ones(2),.1)

    def test_nonpositive_pseudocount_rejected(self):
        with self.assertRaisesRegex(ValueError,'pseudocount'):
            describe(np.ones((2,2)),np.array([0,1]),['a','b'],np.ones(2),0)

    def test_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            describe(np.ones((2,2)),np.array([0,1]),['a'],np.ones(2),.1)

    def test_output_must_not_overlap_inputs(self):
        root=Path(__file__).resolve().parent
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_output(root/'input'/'output',[root/'input'])
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_output(root/'input',[root/'input'/'child.tsv'])


if __name__=='__main__': unittest.main()
