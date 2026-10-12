"""External evaluation-to-summary identities; synthetic fixed arrays, no resampling."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from prediction_extension import workflow
from prediction_extension.boundary import point_replay, save_table, seal
from prediction_extension.evaluation import DEFAULT_MODELS
from prediction_extension.source import COHORTS


class ExternalSummaryIdentityTests(unittest.TestCase):
    def evaluation(self, root, cohort, identity):
        path = root/"inputs"/cohort
        path.mkdir(parents=True)
        ids = [cohort+"__control",cohort+"__crc"]
        meta = pd.DataFrame({"sample_uid":ids,"study_name":[cohort]*2,
                             "study_condition":["control","CRC"]})
        pred = pd.DataFrame([{"sample_uid":sid,"held_out_study":cohort,"panel":panel,
                "classifier":classifier,"k":18,"y":label,"probability_crc":prob}
                for panel,classifier in DEFAULT_MODELS
                for sid,label,prob in zip(ids,[0,1],[.1,.9])])
        metrics = point_replay(pred,meta)
        points = metrics[["auroc","ap"]].to_numpy()
        result = {"metrics":metrics,"model_keys":metrics.model_key.tolist(),
                  "point":points,"bootstrap":np.repeat(points[None,:,:],2000,axis=0),
                  "index_hash":"e"*64}
        workflow.save_evaluation(path,result)
        save_table(path/"predictions.tsv",pred)
        save_table(path/"metadata.tsv",meta)
        details = {"cohort":cohort,"seed":20261011,"n_boot":2000,"bootstrap_context":2,**identity}
        seal(path,"external-evaluate",[],details)
        return path

    def test_external_summary_rejects_mixed_missing_or_mislabeled_identities(self):
        for case in ("mixed_lock","missing_lock","mixed_blind_set","missing_blind_set","wrong_cohort"):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                dirs = []
                for i,cohort in enumerate(COHORTS):
                    identity = {"model_bundle_sha256":"a"*64,"blind_set_sha256":"b"*64}
                    if i==5:
                        if case=="mixed_lock": identity["model_bundle_sha256"]="c"*64
                        elif case=="missing_lock": identity.pop("model_bundle_sha256")
                        elif case=="mixed_blind_set": identity["blind_set_sha256"]="d"*64
                        elif case=="missing_blind_set": identity.pop("blind_set_sha256")
                        elif case=="wrong_cohort": identity["cohort"]="wrong"
                    dirs.append(self.evaluation(root,cohort,identity))
                args = type("Args",(),{"action":"summary-external"})()
                with self.assertRaisesRegex(ValueError,"identity|cohort"):
                    workflow.run(args,{"output_root":root/"result","evaluation_dirs":dirs})
                self.assertFalse((root/"result"/"receipt.json").exists())
                self.assertFalse((root/"result"/"macro_metrics.tsv").exists())

    def test_external_summary_preserves_shared_identity_for_same_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = {"model_bundle_sha256":"a"*64,"blind_set_sha256":"b"*64}
            dirs = [self.evaluation(root,cohort,identity) for cohort in COHORTS]
            args = type("Args",(),{"action":"summary-external"})()
            receipt = workflow.run(args,{"output_root":root/"result","evaluation_dirs":dirs})
            self.assertEqual(receipt["status"],"PASS")
            self.assertEqual(receipt.get("model_bundle_sha256"),"a"*64)
            self.assertEqual(receipt.get("blind_set_sha256"),"b"*64)
            macro = pd.read_csv(root/"result"/"primary3_macro_metrics.tsv",sep="\t")
            self.assertEqual(len(macro),36)
            self.assertTrue(macro["mean"].eq(1.).all())

    def test_blind_identity_is_complete_canonical_and_model_bundle_optional(self):
        self.assertTrue(hasattr(workflow,"_blind_identity"),"blind set identity adapter is missing")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dirs = {}
            for i,cohort in enumerate(COHORTS):
                path = root/cohort
                path.mkdir()
                save_table(path/"predictions_blind.tsv",pd.DataFrame({"sample_id":[cohort+"__sample"]}))
                save_table(path/"sample_vector_hashes.tsv",pd.DataFrame({
                    "sample_id":[cohort+"__sample"],"vector_sha256":[format(i+1,"064x")]}))
                seal(path,"external-predict",[],{"cohort":cohort,"model_bundle_sha256":"a"*64})
                dirs[cohort] = path
            verified = workflow._blind_complete({"blind_dirs":list(dirs.values())})
            identity = workflow._blind_identity(verified)
            self.assertEqual(identity["model_bundle_sha256"],"a"*64)
            self.assertEqual(len(identity["blind_set_sha256"]),64)
            reverse = dict(reversed(list(verified.items())))
            self.assertEqual(workflow._blind_identity(reverse),identity)
            incomplete = dict(verified)
            incomplete.pop(next(iter(incomplete)))
            with self.assertRaises(ValueError):
                workflow._blind_identity(incomplete)
            changed = next(iter(dirs.values()))/"predictions_blind.tsv"
            changed.write_text("sample_id\nchanged\n")
            with self.assertRaises(ValueError):
                workflow._blind_identity(verified)


if __name__=="__main__":
    unittest.main()
