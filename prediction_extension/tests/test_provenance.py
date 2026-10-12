"""Provenance and relocation boundary tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from prediction_extension.boundary import save_table, seal
from prediction_extension.evaluation import DEFAULT_MODELS
from prediction_extension.source import COHORTS
from prediction_extension import workflow


class ProvenanceTests(unittest.TestCase):
    def test_seal_rejects_changed_inputs_since_action_start(self):
        import inspect
        self.assertIn("expected_inputs", inspect.signature(seal).parameters,
                      "sealing lacks protected initial hashes")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"input.tsv"
            source.write_text("value\n1\n")
            from prediction_extension.boundary import sha
            initial = {str(source.resolve()):sha(source)}
            target = root/"output"
            target.mkdir()
            (target/"result.tsv").write_text("value\n1\n")
            source.write_text("value\n2\n")
            with self.assertRaisesRegex(ValueError,"changed"):
                seal(target,"point-replay",[source],expected_inputs=initial)
            self.assertFalse((target/"receipt.json").exists())

    def test_all_blind_cohorts_must_use_same_model_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dirs = []
            for i,cohort in enumerate(COHORTS):
                path = root / cohort
                path.mkdir()
                (path / "predictions_blind.tsv").write_text("sample_id\np1\n")
                seal(path,"external-predict",[],{"cohort":cohort,
                     "model_bundle_sha256":("a"*64 if i<5 else "b"*64)})
                dirs.append(path)
            with self.assertRaisesRegex(ValueError,"lock"):
                workflow._blind_complete({"blind_dirs":dirs})

    def test_complete_blind_grid_rechecks_cross_cohort_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dirs = []
            for i,cohort in enumerate(COHORTS):
                path = root/cohort
                path.mkdir()
                (path/"predictions_blind.tsv").write_text("sample_id\np1\n")
                sid = "repeated" if i<2 else f"sample{i}"
                save_table(path/"sample_vector_hashes.tsv",pd.DataFrame({
                    "sample_id":[sid],"vector_sha256":[str(i)*64]}))
                seal(path,"external-predict",[],{"cohort":cohort,"model_bundle_sha256":"a"*64})
                dirs.append(path)
            with self.assertRaisesRegex(ValueError,"overlap"):
                workflow._blind_complete({"blind_dirs":dirs})

    def test_absolute_launcher_works_from_unrelated_directory_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="prediction outside cwd ") as temp:
            root = Path(temp)
            data = root / "inputs"
            data.mkdir()
            meta = pd.DataFrame({"sample_uid":["a","b"],"study_name":["S","S"],
                                 "study_condition":["control","CRC"]})
            pred = pd.DataFrame([{"sample_uid":sid,"held_out_study":"S","panel":panel,
                 "classifier":classifier,"k":18,"y":label,"probability_crc":prob}
                 for panel,classifier in DEFAULT_MODELS
                 for sid,label,prob in [("a",0,.1),("b",1,.9)]])
            save_table(data/"p.tsv",pred)
            save_table(data/"m.tsv",meta)
            cfg = root/"config with spaces.json"
            cfg.write_text(json.dumps({"predictions":"inputs/p.tsv","metadata":"inputs/m.tsv",
                                       "output_root":"point metrics"}))
            launcher = Path(workflow.__file__).with_name("run_prediction.py")
            self.assertTrue(launcher.is_file(),"absolute-path launcher is missing")
            result = subprocess.run([sys.executable,"-B",str(launcher),"--config",str(cfg),"point-replay"],
                                    cwd=root.parent,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            output = pd.read_csv(root/"point metrics/point_metrics.tsv",sep="\t")
            self.assertEqual(len(output),18)
            self.assertTrue(output.auroc.eq(1.).all())

    def test_summary_rejects_smoke_bootstraps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            from prediction_extension.evaluation import evaluate_study
            from prediction_extension.tests.test_evaluation import cohort
            dirs = []
            names = [f"S{i}" for i in range(9)]
            for i,name in enumerate(names):
                path = root/"inputs"/name
                path.mkdir(parents=True)
                pred,meta = cohort(name)
                workflow.save_evaluation(path,evaluate_study(pred,meta,i,n_boot=2))
                seal(path,"evaluate",[],{"study":name,"seed":20261011,"n_boot":2,"bootstrap_context":1})
                dirs.append(path)
            cfg = {"output_root":root/"result","evaluation_dirs":dirs,"study_names":names}
            args = type("Args",(),{"action":"summary-internal"})()
            with self.assertRaisesRegex(ValueError,"2000"):
                workflow.run(args,cfg)


if __name__ == "__main__":
    unittest.main()
