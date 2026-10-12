"""Synthetic contract tests; no real cohort inputs or fitting."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prediction_extension import external_evaluation as MODULE

PANELS = ["ARP"] + [f"{f}_{d}" for f in ("DCP", "ICP")
                     for d in ("raw_mean", "raw_median", "clr_mean", "clr_median")]
KEYS = sorted(f"{p}::{c}" for p in PANELS for c in ("elastic_net", "random_forest"))
COHORTS = ["C0", "C1", "C2", "C3", "C4", "C5"]
PRIMARY = COHORTS[:3]


def fixture(study="C0"):
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.4, 0.4, 0.7, 0.9])
    metadata = pd.DataFrame({"sample_uid": [f"s{i}" for i in range(6)],
                             "study_name": study,
                             "study_condition": ["CRC" if y else "control" for y in labels]})
    rows = []
    for key in KEYS:
        panel, classifier = key.split("::")
        values = scores if classifier == "elastic_net" else 1 - scores
        for i, (y, score) in enumerate(zip(labels, values)):
            rows.append({"sample_uid": f"s{i}", "held_out_study": study, "panel": panel,
                         "classifier": classifier, "k": 18, "y": y, "probability_crc": score})
    return pd.DataFrame(rows), metadata


def summary_fixture():
    results = {}
    for study, value, n in zip(COHORTS, [0.1, 0.4, 0.7, 0.9, 0.9, 0.9], [6, 60, 600, 6, 6, 6]):
        point = np.full((18, 2), value)
        boots = np.stack([point - .01, point, point + .01])
        ci = np.quantile(boots, [.025, .975], axis=0)
        rows = []
        for j, key in enumerate(KEYS):
            panel, classifier = key.split("::")
            rows.append({"study_name": study, "model_key": key, "panel": panel,
                         "classifier": classifier, "n": n, "n_crc": n // 2,
                         "n_control": n // 2, "prevalence": .5, "k": 18,
                         "auroc": point[j, 0], "ap": point[j, 1],
                         "auroc_ci95_low": ci[0, j, 0], "auroc_ci95_high": ci[1, j, 0],
                         "ap_ci95_low": ci[0, j, 1], "ap_ci95_high": ci[1, j, 1]})
        results[study] = {"model_keys": KEYS.copy(), "point": point, "bootstrap": boots,
                          "metrics": pd.DataFrame(rows), "index_hash": "a" * 64}
    return results


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(MODULE, "external_evaluation.py has not been implemented")
        self.pred, self.metadata = fixture()

    def test_ties_match_sklearn_and_all_duplicate_labels_survive(self):
        result = MODULE.evaluate_cohort(self.pred.sample(frac=1, random_state=4), self.metadata, 0,
                                        n_boot=7)
        self.assertEqual(result["model_keys"], KEYS)
        self.assertEqual(len(result["metrics"]), 18)
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([.1, .4, .4, .4, .7, .9])
        for j, key in enumerate(KEYS):
            score = s if key.endswith("::elastic_net") else 1 - s
            np.testing.assert_allclose(result["point"][j],
                                       [roc_auc_score(y, score), average_precision_score(y, score)])
        for classifier in ("elastic_net", "random_forest"):
            indices = [j for j, key in enumerate(KEYS) if key.endswith("::" + classifier)]
            for j in indices[1:]:
                np.testing.assert_array_equal(result["bootstrap"][:, indices[0]], result["bootstrap"][:, j])

    def test_context_two_seed_shared_resamples_for_distinct_models(self):
        result = MODULE.evaluate_cohort(self.pred, self.metadata, 3, n_boot=9, seed=117)
        rng = np.random.default_rng(np.random.SeedSequence([117, 2, 3]))
        idx = np.concatenate([rng.choice([0, 1, 2], size=(9, 3), replace=True),
                              rng.choice([3, 4, 5], size=(9, 3), replace=True)], axis=1)
        self.assertEqual(result["index_hash"], hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest())
        y = np.array([0, 0, 0, 1, 1, 1])
        for key in ["ARP::elastic_net", "ARP::random_forest"]:
            s = np.array([.1, .4, .4, .4, .7, .9])
            if key.endswith("random_forest"):
                s = 1 - s
            expected = [[roc_auc_score(y[i], s[i]), average_precision_score(y[i], s[i])] for i in idx]
            np.testing.assert_allclose(result["bootstrap"][:, KEYS.index(key)], expected)
        other = MODULE.evaluate_cohort(self.pred, self.metadata, 4, n_boot=9, seed=117)
        self.assertNotEqual(result["index_hash"], other["index_hash"])

    def test_invalid_arguments_fail(self):
        for kwargs in ({"cohort_index": -1}, {"cohort_index": True}, {"cohort_index": 1.2},
                       {"n_boot": 0}, {"n_boot": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MODULE.evaluate_cohort(self.pred, self.metadata, **({"cohort_index": 0, "n_boot": 2} | kwargs))

    def test_invalid_probabilities_labels_grid_and_alignment_fail(self):
        cases = []
        for column, value in [("probability_crc", -0.1), ("probability_crc", np.nan),
                              ("probability_crc", np.inf), ("probability_crc", 1.1),
                              ("y", 2), ("k", 0), ("k", 1.2), ("y", 1)]:
            bad = self.pred.copy()
            bad[column] = bad[column].astype(float)
            bad.loc[0, column] = value
            cases.append(bad)
        cases.extend([self.pred.iloc[1:].copy(), pd.concat([self.pred, self.pred.iloc[:1]]),
                      self.pred.loc[self.pred.panel.ne("ARP")].copy()])
        wrong_study = self.pred.copy()
        wrong_study["held_out_study"] = "other"
        cases.append(wrong_study)
        for i, bad in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                MODULE.evaluate_cohort(bad, self.metadata, 0, n_boot=2)
        with self.assertRaises(ValueError):
            MODULE.evaluate_cohort(self.pred, self.metadata.iloc[:-1], 0, n_boot=2)

    def test_one_class_or_multiple_cohorts_fail(self):
        pred = self.pred.copy()
        meta = self.metadata.copy()
        pred["y"] = 0
        meta["study_condition"] = "control"
        with self.assertRaises(ValueError):
            MODULE.evaluate_cohort(pred, meta, 0, n_boot=2)
        meta.loc[0, "study_name"] = "other"
        pred.loc[pred.sample_uid.eq("s0"), "held_out_study"] = "other"
        with self.assertRaises(ValueError):
            MODULE.evaluate_cohort(pred, meta, 0, n_boot=2)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(MODULE, "external_evaluation.py has not been implemented")
        self.results = summary_fixture()

    def test_sklearn_endpoint_roundoff_preserved_but_substantive_overshoot_fails(self):
        before=np.nextafter(1.,2.)
        self.results['C0']['bootstrap'][0,0,1]=before
        MODULE.summarize_external(self.results,COHORTS,PRIMARY)
        self.assertEqual(self.results['C0']['bootstrap'][0,0,1],before)
        self.results['C0']['bootstrap'][0,0,1]=1.+1e-10
        with self.assertRaises(ValueError):
            MODULE.summarize_external(self.results,COHORTS,PRIMARY)

    def test_primary_three_equal_weight_not_six_or_pooled_and_average_before_quantiles(self):
        macro, contrasts = MODULE.summarize_external(self.results, COHORTS, PRIMARY)
        self.assertEqual(len(macro), 36)
        self.assertEqual(len(contrasts), 574)
        np.testing.assert_allclose(macro["mean"], .4)
        np.testing.assert_allclose(macro["median"], .4)
        np.testing.assert_allclose(macro["minimum"], .1)
        np.testing.assert_allclose(macro["ci95_low"], .3905)
        np.testing.assert_allclose(macro["ci95_high"], .4095)
        self.assertTrue(macro.n_studies.eq(3).all())
        self.assertFalse(np.allclose(macro["mean"], .65))
        self.assertFalse(np.allclose(macro["mean"], (.1 * 6 + .4 * 60 + .7 * 600) / 666))
        self.assertEqual(set(contrasts.scope), set(COHORTS + ["PRIMARY3"]))
        self.assertTrue(macro.inference_status.eq("post_outcome_sensitivity").all())
        self.assertTrue(contrasts.inference_status.eq("post_outcome_sensitivity").all())
        # Cohort bootstrap ranks differ: quantile(mean) != mean(quantile).
        self.results["C1"]["bootstrap"] = self.results["C1"]["bootstrap"][::-1].copy()
        macro, _ = MODULE.summarize_external(self.results, COHORTS, PRIMARY)
        np.testing.assert_allclose(macro.ci95_low, .3968333333333333)

    def test_focal_descriptive_flags_and_nominal_intervals(self):
        _, contrasts = MODULE.summarize_external(self.results, COHORTS, PRIMARY)
        focal = contrasts.loc[contrasts.protocol_focal]
        self.assertEqual(len(focal), 14)
        self.assertEqual(set(focal.contrast), {"ICP_raw_mean_minus_raw_median", "ICP_clr_median_minus_raw_median"})
        self.assertTrue(focal.classifier.eq("elastic_net").all())
        self.assertTrue(focal.metric.eq("AUROC").all())
        self.assertTrue(focal[["ci975_low", "ci975_high"]].notna().all().all())
        self.assertTrue(contrasts.loc[~contrasts.protocol_focal, ["ci975_low", "ci975_high"]].isna().all().all())
        self.assertNotIn("primary", contrasts.columns)

    def test_nonzero_focal_contrast_direction_and_quantiles(self):
        index = KEYS.index("ICP_raw_mean::elastic_net")
        for result in self.results.values():
            result["point"][index, 0] += .05
            result["bootstrap"][:, index, 0] += [.02, .06, .08]
            result["metrics"].loc[result["metrics"].model_key.eq(KEYS[index]), "auroc"] += .05
        _, contrasts = MODULE.summarize_external(self.results, COHORTS, PRIMARY)
        rows = contrasts.loc[(contrasts.contrast == "ICP_raw_mean_minus_raw_median") &
                             (contrasts.classifier == "elastic_net") & (contrasts.metric == "AUROC")]
        np.testing.assert_allclose(rows.estimate, .05)
        np.testing.assert_allclose(rows.ci95_low, .022)
        np.testing.assert_allclose(rows.ci95_high, .079)
        np.testing.assert_allclose(rows.ci975_low, .021)
        np.testing.assert_allclose(rows.ci975_high, .0795)

    def test_missing_non_six_non_three_or_duplicate_cohorts_fail(self):
        for expected, primary in [(COHORTS[:5], PRIMARY), (COHORTS + ["C6"], PRIMARY),
                                  (["C0"] * 6, PRIMARY), (COHORTS, PRIMARY[:2]),
                                  (COHORTS, ["C0"] * 3), (COHORTS, ["C0", "C1", "C9"])]:
            with self.subTest(expected=expected, primary=primary), self.assertRaises(ValueError):
                MODULE.summarize_external(self.results, expected, primary)
        missing = copy.deepcopy(self.results)
        missing.pop("C5")
        with self.assertRaises(ValueError):
            MODULE.summarize_external(missing, COHORTS, PRIMARY)

    def test_arrays_model_order_and_metric_table_mismatches_fail(self):
        mutators = [lambda r: r.update(point=np.zeros((17, 2))),
                    lambda r: r.update(bootstrap=np.zeros((2, 18, 2))),
                    lambda r: r.update(bootstrap=np.zeros((3, 18, 1))),
                    lambda r: r.update(model_keys=KEYS[::-1]),
                    lambda r: r["point"].__setitem__((0, 0), np.nan),
                    lambda r: r["bootstrap"].__setitem__((0, 0, 0), -0.1),
                    lambda r: r.update(metrics=r["metrics"].assign(study_name="wrong")),
                    lambda r: r.update(metrics=r["metrics"].assign(model_key="wrong")),
                    lambda r: r.update(metrics=r["metrics"].assign(auroc=.5)),
                    lambda r: r.update(metrics=r["metrics"].assign(panel="wrong")),
                    lambda r: r.update(metrics=r["metrics"].assign(classifier="wrong"))]
        for i, mutate in enumerate(mutators):
            bad = copy.deepcopy(self.results)
            mutate(bad["C1"])
            with self.subTest(case=i), self.assertRaises(ValueError):
                MODULE.summarize_external(bad, COHORTS, PRIMARY)
        bad = copy.deepcopy(self.results)
        for result in bad.values():
            result["model_keys"][0] = "unexpected::elastic_net"
        with self.assertRaises(ValueError):
            MODULE.summarize_external(bad, COHORTS, PRIMARY)


if __name__ == "__main__":
    unittest.main()
