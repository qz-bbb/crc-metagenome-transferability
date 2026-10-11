"""Behavior tests; synthetic fixtures are not research results."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PATH = Path(__file__).with_name('external_core.py')
core = None
if PATH.exists():
    spec = importlib.util.spec_from_file_location('external_core', PATH)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, 'External validation behavior is not implemented')

    def test_training_match_excludes_diagnosis_conflicts(self):
        old = pd.DataFrame({'sample_uid':['S__a','S__b','S__c'], 'sample_id':['a','b','c'],
                            'study_name':['S']*3, 'study_condition':['CRC','CRC','control'],
                            'subject_id':['a','b','c']})
        new = pd.DataFrame({'Sample Source ID':['a','b','c','new'],
                            'Study condition':['CRC','Control','Control','CRC'],
                            'source_stem':['X']*4})
        kept, audit = core.match_training(old,new)
        self.assertEqual(kept.sample_id.tolist(), ['a','c'])
        self.assertEqual(audit.set_index('sample_id').loc['b','exclusion_reason'], 'label_conflict')
        self.assertNotIn('new', kept.sample_id.tolist())

    def test_ambiguous_sample_mapping_is_not_accepted(self):
        old = pd.DataFrame({'sample_uid':['S__a'], 'sample_id':['a'], 'study_name':['S'],
                            'study_condition':['CRC'], 'subject_id':['a']})
        new = pd.DataFrame({'Sample Source ID':['a','a'], 'Study condition':['CRC','CRC'],
                            'source_stem':['X','Y']})
        with self.assertRaises(ValueError): core.match_training(old,new)

    def test_duplicate_training_subject_is_not_counted_twice(self):
        old = pd.DataFrame({'sample_uid':['S__a','S__b'], 'sample_id':['a','b'],
                            'study_name':['S']*2, 'study_condition':['CRC']*2,
                            'subject_id':['same']*2})
        new = pd.DataFrame({'Sample Source ID':['a','b'], 'Study condition':['CRC']*2,
                            'source_stem':['X']*2})
        kept, audit = core.match_training(old,new)
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit.exclusion_reason.eq('duplicate_subject').sum(), 1)

    def test_training_preprocessing_and_clr_are_fixed(self):
        x = pd.DataFrame([[0.,2.,0.],[2.,2.,0.],[4.,2.,0.]], columns=['a','b','c'])
        state, clr = core.fit_preprocess(x, np.array(['A','B','C']), prevalence=.05, min_studies=2)
        self.assertEqual(state['features'], ['a','b'])
        self.assertEqual(state['pseudocount'], 1.)
        np.testing.assert_allclose(clr[0], [-np.log(3)/2, np.log(3)/2])
        test = pd.DataFrame([[100.,0.,800.]], columns=['b','a','unseen'])
        transformed, coverage = core.transform_external(test, state, same_pipeline=True)
        np.testing.assert_allclose(transformed[0], [-np.log(101)/2,np.log(101)/2])
        self.assertEqual(state['pseudocount'], 1.)

    def test_cross_version_missing_feature_cannot_be_zero_imputed(self):
        state={'features':['a','b'], 'pseudocount':1.}
        with self.assertRaises(ValueError):
            core.transform_external(pd.DataFrame({'a':[1.]}),state,same_pipeline=False)

    def test_within_pipeline_nonreporting_is_explicitly_counted(self):
        state={'features':['a','b'], 'pseudocount':1.}
        x, coverage=core.transform_external(pd.DataFrame({'a':[1.]}),state,same_pipeline=True)
        self.assertEqual(coverage['not_reported_features'], ['b'])
        np.testing.assert_allclose(x[0], [np.log(2)/2,-np.log(2)/2])

    def test_nan_is_not_silently_changed_to_zero(self):
        with self.assertRaises(ValueError):
            core.fit_preprocess(pd.DataFrame({'a':[1.,np.nan]}),np.array(['S','T']),min_studies=1)

    def test_all_ties_and_paired_identical_scores(self):
        y=np.array([0,0,1,1]); p=np.array([.5,.5,.5,.5])
        scores=core.bootstrap_metrics(y,{'a':p,'b':p},100,2)
        self.assertEqual(scores['point']['a'], (.5,.5))
        np.testing.assert_allclose(scores['bootstrap']['a'], np.tile([.5,.5],(100,1)))
        np.testing.assert_array_equal(scores['bootstrap']['a']-scores['bootstrap']['b'],0)

    def test_metric_uncertainty_is_reproducible_and_tie_aware(self):
        y=np.array([0,1,0,1]); p=np.array([0.,.5,.5,1.])
        a=core.bootstrap_metrics(y,{'a':p},30,2)
        b=core.bootstrap_metrics(y,{'a':p},30,2)
        self.assertEqual(a['point']['a'][0], .875)
        np.testing.assert_array_equal(a['bootstrap']['a'], b['bootstrap']['a'])

    def test_random_panel_size_no_overlap_and_determinism(self):
        prev=np.array([.1,.2,.3,.4,.5]); taxa=['a','b','c','d','e']
        a=core.matched_random(taxa,prev,['a','b'],{'a','b'},5)
        b=core.matched_random(taxa,prev,['a','b'],{'a','b'},5)
        self.assertEqual(a,b)
        self.assertEqual(len(set(a)),2)
        self.assertTrue(set(a).isdisjoint({'a','b'}))

    def test_label_join_rejects_missing_participant_and_relabeling(self):
        meta=pd.DataFrame({'Sample Source ID':['a','b'],'Study condition':['CRC','Control']})
        pred=pd.DataFrame({'sample_id':['a','b'],'panel':['P','P'],'classifier':['EN','EN'],
                           'probability_crc':[.7,.2]})
        result=core.attach_labels(pred,meta,{'CRC':1,'Control':1},{'P::EN'})
        self.assertEqual(result['Study condition'].tolist(),['CRC','Control'])
        with self.assertRaises(ValueError): core.attach_labels(pred.iloc[:1],meta,{'CRC':1,'Control':1},{'P::EN'})
        changed=meta.copy(); changed.loc[1,'Study condition']='CRC'
        with self.assertRaises(ValueError): core.attach_labels(pred,changed,{'CRC':1,'Control':1},{'P::EN'})

    def test_incomplete_model_grid_is_rejected(self):
        meta=pd.DataFrame({'Sample Source ID':['a','b'],'Study condition':['CRC','Control']})
        pred=pd.DataFrame({'sample_id':['a','b'],'panel':['P','P'],'classifier':['EN','EN'],
                           'probability_crc':[.7,.2]})
        with self.assertRaises(ValueError): core.attach_labels(pred,meta,{'CRC':1,'Control':1},{'P::EN','P::RF'})

    def test_nonfinite_association_cannot_shrink_bh_family(self):
        with self.assertRaises(ValueError):
            core.assert_finite_statistics(pd.DataFrame({'coef':[1.,1.],'p':[.01,np.nan],'q':[.02,np.nan]}))

if __name__ == '__main__': unittest.main(verbosity=2)

