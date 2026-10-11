import unittest
import numpy as np
import pandas as pd
from identity_core import accession_tokens, resolve_training, assert_disjoint_ids, paired_delta


class IdentityTests(unittest.TestCase):
    def test_only_real_accession_syntax(self):
        self.assertEqual(accession_tokens('ERR1234; SRR55, SAMN123456'), {'ERR1234','SRR55','SAMN123456'})
        self.assertEqual(accession_tokens('unknown; patient12; study2025; nan'), set())

    def test_enriched_direct_and_biosample_not_alias(self):
        train=pd.DataFrame({'sample_uid':['a','b','c','d'], 'sample_id':['alias','alias2','SAMD1234','alias4'],
                            'NCBI_accession':['ERR10','','','notavailable']})
        enriched=pd.DataFrame({'sample_uid':['b','d'], 'run_accession':['SRR20',''], 'sample_accession':['','']})
        got=resolve_training(train,enriched)
        self.assertEqual(got.accession_resolved.tolist(),[True,True,True,False])
        self.assertEqual(got.sample_uid.tolist(),train.sample_uid.tolist())

    def test_duplicate_identity_fails(self):
        train=pd.DataFrame({'sample_uid':['a','a'], 'sample_id':['x','y'], 'NCBI_accession':['ERR1','ERR2']})
        with self.assertRaises(ValueError): resolve_training(train,pd.DataFrame())

    def test_overlap_fails(self):
        with self.assertRaises(ValueError): assert_disjoint_ids(['a','b'],['b','c'])
        assert_disjoint_ids(['a'],['b'])

    def test_paired_difference_direction(self):
        new=np.array([[.7,.6],[.8,.9],[.6,.7]])
        old=np.array([[.6,.5],[.7,.8],[.5,.6]])
        lo,hi=paired_delta(new,old)
        np.testing.assert_allclose(lo,[.1,.1])
        np.testing.assert_allclose(hi,[.1,.1])

if __name__=='__main__': unittest.main()

