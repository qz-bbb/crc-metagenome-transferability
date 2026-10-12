"""CLI and inventory regression tests; tiny fixed predictions only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


class WorkflowTests(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec("prediction_extension.workflow")
        self.assertIsNotNone(spec, "bounded workflow has not been implemented")
        from prediction_extension import workflow
        return workflow

    def test_relocated_point_command_never_refits_or_overwrites(self):
        wf = self.api()
        from prediction_extension.boundary import save_table
        from prediction_extension.evaluation import DEFAULT_MODELS
        with tempfile.TemporaryDirectory(prefix="prediction workflow with spaces ") as temp:
            root = Path(temp)
            inputs = root / "inputs"
            inputs.mkdir()
            meta = pd.DataFrame({"sample_uid": ["a", "b"], "study_name": ["S"]*2,
                                 "study_condition": ["control", "CRC"]})
            pred = pd.DataFrame([{"sample_uid": s, "held_out_study": "S", "panel": p,
                                 "classifier": c, "k": 18, "y": y, "probability_crc": v}
                                for p,c in DEFAULT_MODELS
                                for s,y,v in [("a",0,.1),("b",1,.9)]])
            save_table(inputs / "pred.tsv", pred)
            save_table(inputs / "meta.tsv", meta)
            config = root / "settings.json"
            config.write_text(json.dumps({"predictions": "inputs/pred.tsv", "metadata": "inputs/meta.tsv",
                                         "output_root": "point result"}))
            wf.main(["--config", str(config), "point-replay"])
            saved = pd.read_csv(root / "point result/point_metrics.tsv", sep="\t")
            self.assertEqual(len(saved), 18)
            self.assertTrue(saved.auroc.eq(1.).all())
            with self.assertRaises(FileExistsError):
                wf.main(["--config", str(config), "point-replay"])

    def test_evaluation_checkpoint_roundtrip_preserves_array_order(self):
        wf = self.api()
        from prediction_extension.evaluation import evaluate_study
        from prediction_extension.tests.test_evaluation import cohort
        pred, meta = cohort()
        result = evaluate_study(pred, meta, 0, n_boot=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wf.save_evaluation(root, result)
            restored = wf.load_evaluation(root)
            self.assertEqual(restored["model_keys"], result["model_keys"])
            self.assertEqual(restored["index_hash"], result["index_hash"])
            np.testing.assert_array_equal(restored["point"], result["point"])
            np.testing.assert_array_equal(restored["bootstrap"], result["bootstrap"])

    def test_membership_summary_never_alters_fixed_selection(self):
        wf = self.api()
        from prediction_extension.selection import build_training_state
        from prediction_extension.tests.test_select_fold import training_fixture
        raw, y, studies, ids = training_fixture()
        state = build_training_state(raw,y,studies,ids,{"min_studies":2,"cap":1})
        original = json.dumps(state["memberships"], sort_keys=True)
        tables = wf.membership_summary({"full": state})
        self.assertEqual(len(tables["panel_counts.tsv"]), 9)
        self.assertEqual(len(tables["panel_overlap.tsv"]), 64)
        self.assertIn("training_membership_overlap.tsv", tables)
        self.assertEqual(len(tables["training_membership_overlap.tsv"]),8)
        self.assertTrue(tables["training_membership_overlap.tsv"].k.eq(1).all())
        self.assertEqual(json.dumps(state["memberships"], sort_keys=True), original)

    def test_direction_context_uses_all_features_and_zero_denominator(self):
        wf = self.api()
        raw = pd.DataFrame([[0.,1.],[0.,1.],[0.,2.],[0.,2.]],
                           index=["a","b","c","d"],columns=["t1","t2"])
        evidence = pd.DataFrame({"coef":[1.,1.],"q":[.01,.01],
                                 "eligible_DCP_raw_mean":[True,True]},index=["t1","t2"])
        state = {"features":["t1","t2"],"pseudocount":.5,"evidence":evidence,
                 "memberships":{"ARP":["t1"]},"eligible_counts":{"DCP_raw_mean":2},"k":1}
        summary, details = wf.direction_context(raw,[0,0,1,1],state,"C","primary")
        row = summary[(summary.group=="ALL800") & (summary.external_definition=="raw_mean")].iloc[0]
        self.assertEqual(row.denominator,2)
        self.assertEqual(row.concordant_detected,1)
        self.assertEqual(row.concordance_fraction,.5)
        self.assertEqual(row.not_detected,1)
        self.assertEqual(len(details),8)


if __name__ == "__main__":
    unittest.main()
