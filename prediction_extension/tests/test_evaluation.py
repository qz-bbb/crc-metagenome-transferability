"""Small synthetic contract tests; no real cohort evaluation."""
import hashlib
from io import StringIO
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


from prediction_extension import evaluation as ev

PANELS = ["ARP"] + [f"{family}_{definition}" for family in ("DCP", "ICP")
                     for definition in ("raw_mean", "raw_median", "clr_mean", "clr_median")]
KEYS = sorted(f"{panel}::{clf}" for panel in PANELS
              for clf in ("elastic_net", "random_forest"))


def cohort(study="A", tied=False):
    meta = pd.DataFrame({"sample_uid": [f"{study}{i}" for i in range(6)],
                         "study_name": study,
                         "study_condition": ["control"] * 4 + ["CRC"] * 2})
    scores = [0.1, 0.7, 0.2, 0.8, 0.6, 0.9] if not tied else [0.4] * 6
    rows = [{"sample_uid": uid, "held_out_study": study, "panel": panel,
             "classifier": clf, "k": 3, "y": int(i >= 4),
             "probability_crc": scores[i]}
            for panel in PANELS for clf in ("elastic_net", "random_forest")
            for i, uid in enumerate(meta.sample_uid)]
    return pd.DataFrame(rows), meta


def summary_fixture():
    results = {}
    for i in range(9):
        point = np.tile([0.7, 0.6], (18, 1))
        boots = np.tile(point, (5, 1, 1))
        target = KEYS.index("ICP_raw_mean::elastic_net")
        point[target] = [0.5, 0.4]
        boots[:, target, 0] = [0.3, 0.4, 0.5, 0.6, 0.7]
        boots[:, target, 1] = [0.2, 0.3, 0.4, 0.5, 0.6]
        metrics = pd.DataFrame({"model_key": KEYS, "study_name": f"S{i}",
                                "auroc": point[:, 0], "ap": point[:, 1]})
        results[f"S{i}"] = {"model_keys": KEYS.copy(), "point": point,
                              "bootstrap": boots, "metrics": metrics,
                              "index_hash": str(i)}
    return results


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ev, "Evaluator implementation does not exist yet")

    def test_constant_predictions_retain_all_models_and_standard_tie_metrics(self):
        pred, meta = cohort(tied=True)
        result = ev.evaluate_study(pred, meta, 0, n_boot=3)
        self.assertEqual(result["model_keys"], KEYS)
        self.assertEqual(len(result["metrics"]), 18)
        np.testing.assert_allclose(result["point"], np.tile([0.5, 1 / 3], (18, 1)))
        np.testing.assert_allclose(result["bootstrap"], np.tile([0.5, 1 / 3], (3, 18, 1)))
        self.assertTrue((result["metrics"].n_crc == 2).all())
        self.assertTrue((result["metrics"].n_control == 4).all())

    def test_correct_id_join_makes_input_row_order_irrelevant(self):
        pred, meta = cohort()
        first = ev.evaluate_study(pred, meta, 4, n_boot=7)
        second = ev.evaluate_study(pred.sample(frac=1, random_state=1),
                                   meta.sample(frac=1, random_state=2), 4, n_boot=7)
        pd.testing.assert_frame_equal(first["metrics"], second["metrics"])
        np.testing.assert_array_equal(first["bootstrap"], second["bootstrap"])
        self.assertEqual(first["index_hash"], second["index_hash"])

    def test_nonconstant_standard_metrics_match_hand_ranked_values(self):
        pred, meta = cohort()
        result = ev.evaluate_study(pred, meta, 0, n_boot=3)
        # Positives win six of eight positive-negative pairs. In descending
        # score order positive ranks 1 and 4 give AP=(1+2/4)/2=0.75.
        np.testing.assert_allclose(result["point"], np.tile([0.75, 0.75], (18, 1)))

    def test_paired_resampling_identical_predictions_have_zero_differences(self):
        pred, meta = cohort()
        result = ev.evaluate_study(pred, meta, 2, n_boot=20)
        np.testing.assert_array_equal(result["bootstrap"][:, 1:] - result["bootstrap"][:, :1], 0)

    def test_distinct_models_use_the_same_explicitly_regenerated_indices(self):
        pred, meta = cohort()
        alternative = dict(zip(meta.sample_uid, [0.9, 0.6, 0.3, 0.2, 0.8, 0.1]))
        changed = pred.classifier.eq("random_forest")
        pred.loc[changed, "probability_crc"] = pred.loc[changed, "sample_uid"].map(alternative)
        result = ev.evaluate_study(pred, meta, 5, n_boot=7)
        ordered = meta.sort_values("sample_uid")
        y = ordered.study_condition.map({"control": 0, "CRC": 1}).to_numpy()
        rng = np.random.default_rng(np.random.SeedSequence([20261011, 1, 5]))
        indices = np.concatenate((rng.choice([0, 1, 2, 3], (7, 4), replace=True),
                                  rng.choice([4, 5], (7, 2), replace=True)), axis=1)
        seen_vectors = set()
        for j, key in enumerate(result["model_keys"]):
            panel, classifier = key.split("::")
            scores = pred.loc[pred.panel.eq(panel) & pred.classifier.eq(classifier)]
            probabilities = scores.set_index("sample_uid").loc[
                ordered.sample_uid, "probability_crc"].to_numpy()
            seen_vectors.add(tuple(probabilities))
            for b, idx in enumerate(indices):
                want = [roc_auc_score(y[idx], probabilities[idx]),
                        average_precision_score(y[idx], probabilities[idx])]
                np.testing.assert_array_equal(result["bootstrap"][b, j], want)
        self.assertEqual(len(seen_vectors), 2)
        self.assertFalse(np.array_equal(result["bootstrap"][:, 0], result["bootstrap"][:, 1]))

    def test_seed_indices_have_fixed_class_counts_and_independent_study_streams(self):
        pred, meta = cohort()
        result = ev.evaluate_study(pred, meta, 2, n_boot=7)
        # This is the specified RNG contract, independently checked against the saved hash.
        rng = np.random.default_rng(np.random.SeedSequence([20261011, 1, 2]))
        indices = np.concatenate((rng.choice([0, 1, 2, 3], (7, 4), replace=True),
                                  rng.choice([4, 5], (7, 2), replace=True)), axis=1)
        self.assertTrue(((indices < 4).sum(axis=1) == 4).all())
        want = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
        self.assertEqual(result["index_hash"], want)
        repeated = ev.evaluate_study(pred, meta, 2, n_boot=7)
        other = ev.evaluate_study(pred, meta, 3, n_boot=7)
        self.assertEqual(result["index_hash"], repeated["index_hash"])
        self.assertNotEqual(result["index_hash"], other["index_hash"])

    def test_missing_duplicate_extra_samples_and_models_fail(self):
        pred, meta = cohort()
        bad_cases = [pred.iloc[1:], pd.concat([pred, pred.iloc[:1]]),
                     pred.assign(sample_uid=lambda x: x.sample_uid.mask(x.index == 0, "extra")),
                     pred.assign(panel=lambda x: x.panel.mask(x.index == 0, "extra"))]
        for bad in bad_cases:
            with self.subTest(case=len(bad)), self.assertRaises(ValueError):
                ev.validate_predictions(bad, meta)
        with self.assertRaises(ValueError):
            ev.validate_predictions(pred, pd.concat([meta, meta.iloc[:1]]))

    def test_label_group_k_and_probability_conflicts_fail(self):
        pred, meta = cohort()
        for column, value in [("y", 1), ("held_out_study", "wrong"), ("k", 4),
                              ("probability_crc", np.nan), ("probability_crc", np.inf),
                              ("probability_crc", -0.01), ("probability_crc", 1.01)]:
            bad = pred.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                ev.validate_predictions(bad, meta)
        with self.assertRaises(ValueError):
            ev.validate_predictions(pred, meta.assign(study_condition="case"))

    def test_explicit_smaller_model_grid_and_single_fold_metadata_work(self):
        pred, meta = cohort()
        selected = pred.loc[(pred.panel == "ARP") & (pred.classifier == "elastic_net")]
        aligned = ev.validate_predictions(selected, meta,
                                          expected_models=[("ARP", "elastic_net")])
        self.assertEqual(len(aligned), 6)
        self.assertEqual(aligned.sample_uid.tolist(), meta.sample_uid.tolist())

    def test_one_class_and_invalid_bootstrap_counts_fail(self):
        pred, meta = cohort()
        with self.assertRaises(ValueError):
            ev.evaluate_study(pred.assign(y=0), meta.assign(study_condition="control"), 0, n_boot=3)
        for n_boot in (0, -1, 1.5):
            with self.subTest(n_boot=n_boot), self.assertRaises(ValueError):
                ev.evaluate_study(pred, meta, 0, n_boot=n_boot)


class SummaryContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ev, "Evaluator implementation does not exist yet")

    def test_nine_fold_missing_extra_divergent_grid_or_shape_rejected(self):
        results = summary_fixture()
        expected = list(results)
        for bad in ({k: v for k, v in results.items() if k != "S0"},
                    dict(results, extra=results["S0"])):
            with self.assertRaises(ValueError):
                ev.summarize_studies(bad, expected)
        for field, value in [("model_keys", KEYS[::-1]),
                              ("bootstrap", np.zeros((4, 18, 2)))]:
            bad = summary_fixture()
            bad["S0"][field] = value
            with self.assertRaises(ValueError):
                ev.summarize_studies(bad, expected)
        with self.assertRaises(ValueError):
            ev.summarize_studies(results, expected[:8])

    def test_checkpoint_csv_roundtrip_accepts_float_precision_only(self):
        results = summary_fixture()
        score = np.nextafter(1 / 3, 1)
        result = results["S0"]
        result["point"][:, 0] = score
        result["metrics"]["auroc"] = score
        result["metrics"] = pd.read_csv(StringIO(result["metrics"].to_csv(index=False)))
        try:
            macro, _ = ev.summarize_studies(results, list(results))
        except ValueError as exc:
            self.fail(f"CSV floating precision alone must not block a valid checkpoint: {exc}")
        self.assertEqual(len(macro), 36)
        result["metrics"].loc[0, "auroc"] += 0.01
        with self.assertRaises(ValueError):
            ev.summarize_studies(results, list(results))

    def test_macro_is_equal_fold_mean_not_pooled_and_ci_means_before_quantiles(self):
        results = summary_fixture()
        # Equal-size two-class folds can have perfect within-study AUROC while
        # incomparable offsets reduce pooled AUROC (0.6,0.7 and 0.1,0.2).
        # Nine perfect fold points therefore must yield 1, never a pooled score.
        for i, result in enumerate(results.values()):
            result["point"][:, 0] = 1.0
            result["metrics"]["auroc"] = 1.0
            result["bootstrap"][:, :, 0] = np.array([0, 0.25, 0.5, 0.75, 1])[:, None]
            if i >= 4:
                result["bootstrap"][:, :, 0] = result["bootstrap"][::-1, :, 0]
        macro, _ = ev.summarize_studies(results, list(results))
        row = macro.loc[(macro.model_key == KEYS[0]) & (macro.metric == "AUROC")].iloc[0]
        self.assertEqual(row["mean"], 1)
        self.assertEqual(row["median"], 1)
        self.assertEqual(row["minimum"], 1)
        self.assertAlmostEqual(row.ci95_low, 4.025 / 9)
        self.assertAlmostEqual(row.ci95_high, 4.975 / 9)

    def test_actual_equal_size_folds_macro_differs_from_pooled_metric(self):
        results, pooled_y, pooled_scores = {}, [], []
        for i in range(9):
            pred, meta = cohort(f"S{i}")
            scores = np.array([0.1] * 4 + [0.2] * 2) + (0.6 if i >= 4 else 0)
            by_uid = dict(zip(meta.sample_uid, scores))
            pred["probability_crc"] = pred.sample_uid.map(by_uid)
            results[f"S{i}"] = ev.evaluate_study(pred, meta, i, n_boot=3)
            pooled_y.extend([0] * 4 + [1] * 2)
            pooled_scores.extend(scores)
        macro, _ = ev.summarize_studies(results, list(results))
        self.assertTrue((macro.loc[macro.metric.eq("AUROC"), "mean"] == 1).all())
        self.assertAlmostEqual(roc_auc_score(pooled_y, pooled_scores), 61 / 81)

    def test_primary_literal_quantiles_negative_results_and_ap_supporting_only(self):
        results = summary_fixture()
        _, contrasts = ev.summarize_studies(results, list(results))
        rows = contrasts.loc[(contrasts.scope == "MACRO") &
                             (contrasts.contrast == "ICP_raw_mean_minus_raw_median") &
                             (contrasts.classifier == "elastic_net")]
        auroc = rows.loc[rows.metric == "AUROC"].iloc[0]
        self.assertAlmostEqual(auroc.estimate, -0.2)
        self.assertAlmostEqual(auroc.ci95_low, -0.39)
        self.assertAlmostEqual(auroc.ci95_high, -0.01)
        self.assertAlmostEqual(auroc.ci975_low, -0.395)
        self.assertAlmostEqual(auroc.ci975_high, -0.005)
        self.assertTrue(auroc.primary)
        ap = rows.loc[rows.metric == "AP"].iloc[0]
        self.assertFalse(ap.primary)
        self.assertTrue(pd.isna(ap.ci975_low))
        self.assertEqual(len(contrasts), 820)
        self.assertEqual(int(contrasts.primary.sum()), 20)

    def test_linear_interaction_and_rf_minus_en_have_expected_direction(self):
        results = summary_fixture()
        for result in results.values():
            changes = {"DCP_clr_mean::elastic_net": 0.8,
                       "DCP_clr_median::elastic_net": 0.4,
                       "DCP_raw_mean::elastic_net": 0.6,
                       "DCP_raw_median::elastic_net": 0.5,
                       "ARP::random_forest": 0.3}
            for key, score in changes.items():
                result["point"][KEYS.index(key), 0] = score
                result["metrics"].loc[result["metrics"].model_key == key, "auroc"] = score
                result["bootstrap"][:, KEYS.index(key), 0] = score
        _, contrasts = ev.summarize_studies(results, list(results))
        rows = contrasts.loc[(contrasts.scope == "MACRO") & (contrasts.metric == "AUROC")]
        interaction = rows.loc[(rows.contrast == "DCP_interaction") &
                               (rows.classifier == "elastic_net")].iloc[0]
        self.assertAlmostEqual(interaction.estimate, 0.3)
        self.assertAlmostEqual(interaction.ci95_low, 0.3)
        rf = rows.loc[rows.contrast == "ARP_RF_minus_EN"].iloc[0]
        self.assertAlmostEqual(rf.estimate, -0.4)


if __name__ == "__main__":
    unittest.main()
