"""Corrupted derived state must not reuse a frozen identity or reach fitting."""
import copy
import sys
import unittest
from pathlib import Path

import prediction_extension.models as run_fold
import prediction_extension.selection as select_fold
from prediction_extension.tests.test_select_fold import training_fixture


class StateIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, labels, study, ids = training_fixture()
        split = select_fold.split_outer_fold(raw, labels, study, ids, 'study_c')
        cls.train, cls.test = split['train'], split['test']
        cls.state = select_fold.build_training_state(cls.train['raw'], cls.train['y'], cls.train['studies'], cls.train['sample_ids'], {'min_studies': 2, 'cap': 1})

    def assert_consumers_reject(self, state):
        consumers = {
            'model_fingerprint': lambda: run_fold.model_fingerprint(state, 'elastic_net', 'ARP', None),
            'fit_panel': lambda: run_fold.fit_panel(state, self.train['raw'], self.train['y'], self.test['raw'], 'elastic_net', 'ARP', None),
        }
        for name, call in consumers.items():
            with self.subTest(consumer=name), self.assertRaises(ValueError):
                call()

    def test_pseudocount_mutation_is_rejected_before_fingerprint_and_fit(self):
        state = copy.deepcopy(self.state); state['pseudocount'] *= 10
        self.assert_consumers_reject(state)

    def test_retained_order_mutation_is_rejected_before_fingerprint_and_fit(self):
        state = copy.deepcopy(self.state); state['features'].reverse()
        self.assert_consumers_reject(state)

    def test_membership_mutation_is_rejected_before_fingerprint_and_fit(self):
        state = copy.deepcopy(self.state)
        state['memberships']['ARP'] = [taxon for taxon in state['features'] if taxon not in state['memberships']['ARP']][:1]
        self.assert_consumers_reject(state)

    def test_config_mutation_is_rejected_before_fingerprint_and_fit(self):
        state = copy.deepcopy(self.state); state['config']['cap'] = 2
        self.assert_consumers_reject(state)

    def test_verifier_accepts_untouched_real_built_state_and_preserves_identity(self):
        verifier = getattr(select_fold, 'verify_training_state_integrity', None)
        self.assertTrue(callable(verifier), 'Derived state identity verifier is missing')
        self.assertEqual(verifier(self.state), self.state['hashes']['training_state'])
        before = run_fold.model_fingerprint(self.state, 'elastic_net', 'ARP', None)
        verifier(self.state)
        self.assertEqual(run_fold.model_fingerprint(self.state, 'elastic_net', 'ARP', None), before)

    def test_untouched_built_state_still_fits(self):
        probability = run_fold.fit_panel(self.state, self.train['raw'], self.train['y'], self.test['raw'], 'elastic_net', 'ARP', None)
        self.assertEqual(probability.shape, (24,))
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())


if __name__ == '__main__':
    unittest.main()
