"""Hand-checked software fixtures; these are not biological validation."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np



class CoreTests(unittest.TestCase):
    def setUp(self):
        try:
            self.core = importlib.import_module('prediction_extension.core')
        except ModuleNotFoundError:
            self.fail('definition_prediction_core implementation is missing')

    def test_raw_median_zero_is_distinct_from_mean_and_detection(self):
        raw = np.array([[0, 1], [0, 1], [1, 1], [0, 1], [0, 1], [.1, 1]])
        effects, detected = self.core.effect_definitions(raw, [1, 1, 1, 0, 0, 0], .05)
        self.assertEqual(set(effects), {'raw_mean', 'raw_median', 'clr_mean', 'clr_median'})
        self.assertEqual(effects['raw_median'][0], 0.)
        self.assertAlmostEqual(effects['raw_mean'][0], .3)
        self.assertTrue(detected[0])

    def test_clr_uses_complete_feature_denominator(self):
        np.testing.assert_allclose(self.core.clr([[1, 3]], 1), [[-.34657359027997265, .34657359027997265]])

    def test_direction_zero_and_undetected_remain_denominator(self):
        concordant = self.core.concordance([0, 2, -3, 1e-13, 1], [1, 1, 1, 1, 0], [1, 0, 1, 1, 1])
        np.testing.assert_array_equal(concordant, [0, 0, 0, 0, 0])
        np.testing.assert_allclose(self.core.agreement([[1, 0], [0, 0], [0, 1]]), [1 / 3, 1 / 3])

    def test_nonfinite_reference_sign_is_rejected(self):
        with self.assertRaises(ValueError):
            self.core.concordance([1], [np.nan], [True])

    def test_common_size_includes_smallest_and_zero(self):
        groups = {i: set(range(20)) for i in range(8)}
        groups[3] = set(range(14))
        self.assertEqual(self.core.common_panel_size(groups), 14)
        groups[3] = set()
        self.assertEqual(self.core.common_panel_size(groups), 0)
        with self.assertRaises(ValueError):
            self.core.common_panel_size({i: {1} for i in range(7)})

    def test_paired_ties_have_exact_zero_differences_and_class_proportion_ap(self):
        rows = self.core.paired_metrics([0, 0, 0, 1], {'a': [.5] * 4, 'b': [.5] * 4}, [[0, 1, 2, 3], [0, 3, 3, 3]])
        self.assertEqual(len(rows), 4)
        self.assertTrue((rows.difference == 0).all())
        np.testing.assert_array_equal(rows.loc[rows.metric == 'AUROC', 'left_value'], [.5, .5])
        np.testing.assert_array_equal(rows.loc[rows.metric == 'AP', 'left_value'], [.25, .75])
        self.assertEqual(rows.contrast.unique().tolist(), ['a - b'])

    def test_three_models_compare_all_pairs_on_shared_replicates(self):
        rows = self.core.paired_metrics([0, 0, 1, 1], {'a': [0, 0, 1, 1], 'b': [.5] * 4, 'c': [1, 1, 0, 0]}, [[0, 1, 2, 3]])
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows.loc[(rows.left == 'a') & (rows.right == 'c') & (rows.metric == 'AUROC'), 'difference'].item(), 1.)

    def test_invalid_metric_inputs_fail_closed(self):
        for labels, predictions, indices in [
            ([0, 1], {'a': [0, 2], 'b': [.5, .5]}, [[0, 1]]),
            ([0, 1], {'a': [0, np.nan], 'b': [.5, .5]}, [[0, 1]]),
            ([0, 1], {'a': [0, 1], 'b': [.5, .5]}, [[0, 0]]),
            ([0, 1], {'a': [[0], [1]], 'b': [.5, .5]}, [[0, 1]]),
            ([0, 2], {'a': [0, 1], 'b': [.5, .5]}, [[0, 1]]),
            ([0, 1], {'a': [0, 1], 'b': [.5, .5]}, [[0, .5]]),
        ]:
            with self.subTest(labels=labels, indices=indices), self.assertRaises(ValueError):
                self.core.paired_metrics(labels, predictions, indices)

    def test_invalid_effect_inputs_fail_closed(self):
        for raw, labels, pseudo in [([[np.nan, 1], [1, 1]], [0, 1], 1), ([[1, 1], [1, 1]], [0, 0], 1), ([[1, 1], [1, 1]], [0, 1], 0), ([[-1, 1], [1, 1]], [0, 1], 1)]:
            with self.subTest(raw=raw, labels=labels), self.assertRaises(ValueError):
                self.core.effect_definitions(raw, labels, pseudo)


if __name__ == '__main__':
    unittest.main()
