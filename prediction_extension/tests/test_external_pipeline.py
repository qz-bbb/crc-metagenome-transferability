import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
try:
    import prediction_extension.external as ep
except ModuleNotFoundError:
    ep = None


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ep, 'external_pipeline must exist')

    def test_same_pipeline_missing_rows_only(self):
        raw = pd.DataFrame({'a': [1., 2.]}, index=['e1', 'e2'])
        state = {'features': ['a', 'b'], 'pseudocount': .01}
        with self.assertRaises(ValueError):
            ep.transform_blind(raw, state, same_source_verified=False)
        x, missing = ep.transform_blind(raw, state, same_source_verified=True)
        expected = np.log([[1.01, .01], [2.01, .01]])
        expected -= expected.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(x, expected)
        self.assertEqual(missing, ['b'])

    def test_transform_rejects_invalid_input(self):
        for raw in [pd.DataFrame({'a': [-1.]}), pd.DataFrame({'a': [np.nan]}),
                    pd.DataFrame([[1., 2.]], columns=['a', 'a']), pd.DataFrame({'a': []})]:
            with self.assertRaises(ValueError):
                ep.transform_blind(raw, {'features': ['a'], 'pseudocount': .01}, True)

    def test_label_attachment_keeps_adenoma_out_of_control(self):
        blind = pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'panel': ['ARP']*3,
            'classifier': ['elastic_net']*3, 'k': [18]*3, 'probability_crc': [.7, .3, .6]})
        meta = pd.DataFrame({'Sample Source ID': ['c', 'b', 'a'],
                             'Study condition': ['Adenoma', 'Control', 'CRC']})
        p, m = ep.attach_outcomes(blind, meta, 'X', {'CRC':1,'Control':1,'Adenoma':1},
                                  expected_models=[('ARP','elastic_net')])
        self.assertEqual(set(p.sample_id), {'a', 'b'})
        self.assertEqual(dict(zip(p.sample_id, p.y)), {'a':1, 'b':0})
        self.assertEqual(set(m.study_condition), {'CRC','control'})
        self.assertEqual(p.sample_uid.tolist(), ['X__a', 'X__b'])

    def test_missing_samples_and_changed_labels_fail(self):
        blind = pd.DataFrame({'sample_id': ['a'], 'panel':['ARP'], 'classifier':['elastic_net'],
                              'k':[18], 'probability_crc':[.6]})
        for meta in [pd.DataFrame({'Sample Source ID':['b'],'Study condition':['CRC']}),
                     pd.DataFrame({'Sample Source ID':['a'],'Study condition':['Unknown']})]:
            with self.assertRaises(ValueError):
                ep.attach_outcomes(blind, meta, 'X', {'CRC':1}, [('ARP','elastic_net')])

    def test_historical_replay_checks_identity_and_tolerance(self):
        new = pd.DataFrame([{'sample_id': sid, 'panel': p, 'classifier': c, 'probability_crc': v}
            for p in ['ARP','DCP_raw_median','ICP_raw_median']
            for c in ['elastic_net','random_forest'] for sid,v in [('a',.3),('b',.8)]])
        old = new.replace({'panel': ep.BASELINE_PANELS, 'classifier': ep.BASELINE_CLASSIFIERS})
        checks = ep.compare_baseline(new, old)
        self.assertEqual(len(checks), 6)
        self.assertTrue(checks['pass'].all())
        old.loc[0, 'probability_crc'] += .01
        with self.assertRaises(ValueError):
            ep.compare_baseline(new, old)
        with self.assertRaises(ValueError):
            ep.compare_baseline(new, old.iloc[1:])

    def test_fitting_api_cannot_accept_external_data(self):
        import inspect
        self.assertEqual(list(inspect.signature(ep.fit_locked).parameters), ['state','raw','y'])


if __name__ == '__main__':
    unittest.main()
