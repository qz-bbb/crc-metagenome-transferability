import importlib
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from prediction_extension.tests.test_select_fold import training_fixture
from prediction_extension.selection import build_training_state, split_outer_fold


class FittingTests(unittest.TestCase):
    def setUp(self):
        try:
            self.fit = importlib.import_module('prediction_extension.models')
        except ModuleNotFoundError:
            self.fail('run_fold implementation is missing')
        raw, y, study, ids = training_fixture()
        split = split_outer_fold(raw, y, study, ids, 'study_c')
        self.train, self.test = split['train'], split['test']
        self.state = build_training_state(self.train['raw'], self.train['y'], self.train['studies'], self.train['sample_ids'], {'min_studies': 2, 'cap': 1})

    def test_complete_grid_probabilities_determinism_and_fingerprint(self):
        receipts = []
        for label in self.state['memberships']:
            family, definition = ('ARP', None) if label == 'ARP' else label.split('_', 1)
            for classifier in ['elastic_net', 'random_forest']:
                args = (self.state, self.train['raw'], self.train['y'], self.test['raw'], classifier, family, definition)
                p = self.fit.fit_panel(*args)
                again = self.fit.fit_panel(*args)
                self.assertEqual(p.shape, (24,))
                self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
                np.testing.assert_allclose(p, again, atol=1e-10, rtol=1e-8)
                receipts.append(self.fit.model_fingerprint(self.state, classifier, family, definition))
        self.assertEqual(len(receipts), 18)
        self.assertEqual(len(set(receipts)), 18)

    def test_selected_predictors_use_all_retained_clr_denominator(self):
        # Altering a retained but unselected taxon must alter selected CLR coordinates.
        original, _ = self.fit.panel_predictors(self.state, self.train['raw'], self.test['raw'], 'ARP', None)
        other = next(taxon for taxon in self.state['features'] if taxon not in self.state['memberships']['ARP'])
        changed = self.train['raw'].copy(); changed.loc[:, other] *= 1e4
        modified, _ = self.fit.panel_predictors(self.state, changed, self.test['raw'], 'ARP', None)
        self.assertFalse(np.allclose(original, modified))
        selected = self.state['memberships']['ARP'][0]
        raw_row = self.train['raw'].loc[:, self.state['features']].iloc[0].to_numpy()
        logs = np.log(raw_row + self.state['pseudocount'])
        expected = logs[self.state['features'].index(selected)] - logs.mean()
        self.assertAlmostEqual(original[0, 0], expected)

    def test_missing_retained_test_feature_is_not_zero_filled(self):
        absent = self.test['raw'].drop(columns=self.state['features'][-1])
        with self.assertRaises(ValueError):
            self.fit.fit_panel(self.state, self.train['raw'], self.train['y'], absent, 'elastic_net', 'ARP', None)

    def test_changed_train_data_cannot_reuse_frozen_state(self):
        changed = self.train['raw'].copy(); changed.iloc[0, 0] *= 1.01
        with self.assertRaises(ValueError):
            self.fit.fit_panel(self.state, changed, self.train['y'], self.test['raw'], 'elastic_net', 'ARP', None)

    def test_empty_panel_reports_unavailable(self):
        state = build_training_state(self.train['raw'], self.train['y'], self.train['studies'], self.train['sample_ids'], {'min_studies': 2, 'cap': 0})
        with self.assertRaises(self.fit.PanelUnavailableError):
            self.fit.fit_panel(state, self.train['raw'], self.train['y'], self.test['raw'], 'elastic_net', 'ARP', None)

    def test_convergence_warning_is_not_silently_swallowed_or_tuned(self):
        def nonconvergent_fit(model, predictors, labels):
            warnings.warn('controlled numerical nonconvergence', ConvergenceWarning)
        with patch.object(self.fit.LogisticRegression, 'fit', nonconvergent_fit):
            with self.assertRaises(self.fit.ModelConvergenceError):
                self.fit.fit_panel(self.state, self.train['raw'], self.train['y'], self.test['raw'], 'elastic_net', 'ARP', None)

    def test_scaling_is_train_only_for_both_classifiers(self):
        # RF is insensitive to monotone scaling, so observe the actual trained pipeline boundary.
        for classifier in ['elastic_net', 'random_forest']:
            pipeline = self.fit.make_model(classifier)
            pipeline.fit(np.array([[0., 1.], [2., 4.], [4., 8.], [6., 16.]]), [0, 0, 1, 1])
            scaler = pipeline.named_steps['scaler']
            np.testing.assert_allclose(scaler.mean_, [3., 7.25])
            before = scaler.mean_.copy(); pipeline.predict_proba([[1e6, 1e7]])
            np.testing.assert_array_equal(scaler.mean_, before)


if __name__ == '__main__':
    unittest.main()
