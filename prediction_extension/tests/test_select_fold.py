import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd



def training_fixture():
    rng = np.random.default_rng(827)
    studies = np.repeat(['study_a', 'study_b', 'study_c'], 24)
    y = np.tile(np.repeat([0, 1], 12), 3)
    logs = rng.normal(0, .15, (72, 4)) + y[:, None] * [2, -2, 0, 0]
    raw = pd.DataFrame(np.exp(logs), columns=['taxon_d', 'taxon_b', 'taxon_a', 'taxon_c'], index=[f's{i}' for i in range(72)])
    return raw, y, studies, raw.index.tolist()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.select = importlib.import_module('prediction_extension.selection')
        except ModuleNotFoundError:
            self.fail('select_fold implementation is missing')
        self.raw, self.y, self.studies, self.ids = training_fixture()

    def state(self, raw=None, labels=None, studies=None, ids=None, config=None):
        return self.select.build_training_state(self.raw if raw is None else raw, self.y if labels is None else labels, self.studies if studies is None else studies, self.ids if ids is None else ids, {'min_studies': 2} if config is None else config)

    def test_state_keeps_complete_grid_and_full_statistics_family(self):
        state = self.state()
        self.assertEqual(state['features'], ['taxon_a', 'taxon_b', 'taxon_c', 'taxon_d'])
        self.assertEqual(len(state['memberships']), 9)
        self.assertEqual(len(state['eligible_counts']), 8)
        self.assertGreater(state['k'], 0)
        self.assertTrue(all(len(v) == state['k'] for v in state['memberships'].values()))
        self.assertEqual(state['evidence'].index.tolist(), state['features'])
        self.assertTrue(np.isfinite(state['evidence'].select_dtypes(include=[np.number])).all().all())
        self.assertEqual(len(state['direction_details']), 3 * 4 * 4)
        self.assertTrue({'effect', 'exact_zero', 'tolerance_zero', 'detected', 'same_direction', 'reverse_direction'}.issubset(state['direction_details'].columns))
        self.assertEqual(state['pseudocount'], self.raw.to_numpy().min() / 2)

    def test_der_simonian_laird_hand_fixture(self):
        p, i2 = self.select.random_effects(np.array([[1., 2.], [1., 2.]]), np.ones((2, 2)))
        np.testing.assert_allclose(p, [.157299207050285, .004677734981047266], rtol=1e-12)
        np.testing.assert_array_equal(i2, [0., 0.])

    def test_bh_never_silently_shrinks_invalid_family(self):
        with self.assertRaises(ValueError):
            self.select.bh([.01, np.nan])
        np.testing.assert_allclose(self.select.bh([.01, .04, .2]), [.03, .06, .2])

    def test_duplicate_and_invalid_training_boundaries(self):
        bad_column = self.raw.copy(); bad_column.columns = ['a', 'a', 'b', 'c']
        bad_ids = self.ids.copy(); bad_ids[1] = bad_ids[0]
        nonfinite = self.raw.copy(); nonfinite.iloc[0, 0] = np.nan
        zeros = self.raw * 0
        one_class_study = self.y.copy(); one_class_study[self.studies == 'study_a'] = 0
        for raw, labels, ids in [(bad_column, self.y, self.ids), (self.raw, self.y, bad_ids), (nonfinite, self.y, self.ids), (zeros, self.y, self.ids), (self.raw, one_class_study, self.ids)]:
            with self.subTest(ids=ids[:2]), self.assertRaises(ValueError):
                self.state(raw=raw, labels=labels, ids=ids)

    def test_actual_outer_split_isolates_heldout_values_labels_and_only_taxa(self):
        baseline = self.select.split_outer_fold(self.raw, self.y, self.studies, self.ids, 'study_c')
        altered = self.raw.copy(); altered.loc[self.studies == 'study_c'] *= 1e12
        altered['heldout_only'] = np.where(self.studies == 'study_c', 1e14, 0.)
        changed_labels = self.y.copy(); changed_labels[self.studies == 'study_c'] = 1 - changed_labels[self.studies == 'study_c']
        comparison = self.select.split_outer_fold(altered, changed_labels, self.studies, self.ids, 'study_c')
        states = [self.select.build_training_state(part['train']['raw'], part['train']['y'], part['train']['studies'], part['train']['sample_ids'], {'min_studies': 2}) for part in [baseline, comparison]]
        self.assertEqual(states[0]['hashes'], states[1]['hashes'])
        self.assertEqual(states[0]['memberships'], states[1]['memberships'])
        self.assertTrue(set(baseline['train']['sample_ids']).isdisjoint(baseline['test']['sample_ids']))
        self.assertFalse(np.array_equal(baseline['test']['y'], comparison['test']['y']))

    def test_fingerprints_change_for_training_contents_labels_studies_ids_config(self):
        baseline = self.state()['hashes']['training_state']
        altered_raw = self.raw.copy(); altered_raw.iloc[0, 0] *= 1.01
        altered_y = self.y.copy(); altered_y[[0, 12]] = altered_y[[12, 0]]
        altered_studies = np.char.add(self.studies.astype(str), '_renamed')
        altered_ids = ['x_' + value for value in self.ids]; renamed_raw = self.raw.copy(); renamed_raw.index = altered_ids
        variations = [self.state(raw=altered_raw), self.state(labels=altered_y), self.state(studies=altered_studies), self.state(raw=renamed_raw, ids=altered_ids), self.state(config={'min_studies': 2, 'cap': 1})]
        self.assertTrue(all(state['hashes']['training_state'] != baseline for state in variations))
        self.assertEqual(self.state()['hashes']['training_state'], baseline)
        self.assertTrue({'numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn'}.issubset(self.state()['package_versions']))

    def test_rank_deficient_association_fails_closed(self):
        with self.assertRaises(ValueError):
            self.select.association(np.array([[1], [2], [3], [4]], float), [0, 0, 1, 1], ['a', 'a', 'b', 'b'])

    def test_inner_sign_retention_keeps_tiny_nonzero_same_sign_coefficients(self):
        self.assertTrue(callable(getattr(self.select, 'coefficient_sign_concordance', None)), 'Coefficient-sign primitive is missing')
        np.testing.assert_array_equal(self.select.coefficient_sign_concordance([1e-15, -1e-15, 0], [1, -1, 1]), [True, True, False])


if __name__ == '__main__':
    unittest.main()
