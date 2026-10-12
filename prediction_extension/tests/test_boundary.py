"""Boundary tests: reject unsafe writes, lossy inputs and incomplete grids."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


class BoundaryTests(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec("prediction_extension.boundary")
        self.assertIsNotNone(spec, "portable boundary API has not been implemented")
        from prediction_extension import boundary
        return boundary

    def test_config_paths_follow_config_after_cwd_change(self):
        api = self.api()
        with tempfile.TemporaryDirectory(prefix="prediction relocated ") as temp:
            root = Path(temp)
            config = root / "local settings.json"
            config.write_text(json.dumps({"source_root": "inputs", "training_manifest": "inputs/train.tsv",
                                         "output_root": "results"}))
            cwd = Path.cwd()
            try:
                os.chdir(root.parent)
                value = api.load_config(config)
            finally:
                os.chdir(cwd)
            self.assertEqual(value["source_root"], root / "inputs")
            self.assertEqual(value["output_root"], root / "results")

    def test_source_output_overlap_is_rejected_in_both_directions(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for source, output in [(root, root), (root, root / "out"), (root / "inputs", root)]:
                with self.subTest(source=source, output=output):
                    with self.assertRaises(ValueError):
                        api.validate_output(output, [source])

    def test_code_overlap_and_existing_destination_are_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                api.validate_output(Path(api.__file__).parent / "result", [])
            output = root / "result"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                api.fresh(output, [root / "inputs"])

    def test_duplicate_column_and_lossless_float_read(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.tsv"
            path.write_text("sample_uid\tp\np1\t0.10000000000000003\n")
            self.assertEqual(api.read_tsv(path).p.iloc[0], 0.10000000000000003)
            path.write_text("p\tp\n1\t2\n")
            with self.assertRaises(ValueError):
                api.read_tsv(path)

    def test_point_replay_requires_every_panel_by_learner(self):
        api = self.api()
        from prediction_extension.evaluation import DEFAULT_MODELS
        metadata = pd.DataFrame({"sample_uid": ["a", "b", "c", "d"],
                                 "study_name": ["study"] * 4,
                                 "study_condition": ["control", "control", "CRC", "CRC"]})
        rows = [{"sample_uid": sid, "held_out_study": "study", "panel": panel,
                 "classifier": clf, "k": 18, "y": label, "probability_crc": prob}
                for panel, clf in DEFAULT_MODELS
                for sid, label, prob in zip(["a", "b", "c", "d"], [0, 0, 1, 1], [.1, .2, .8, .9])]
        pred = pd.DataFrame(rows)
        result = api.point_replay(pred, metadata)
        self.assertEqual(len(result), 18)
        self.assertTrue(result.auroc.eq(1.).all())
        self.assertTrue(result.ap.eq(1.).all())
        with self.assertRaises(ValueError):
            api.point_replay(pred.iloc[:-1], metadata)
        duplicate = pd.concat([metadata, metadata.iloc[:1]])
        with self.assertRaises(ValueError):
            api.point_replay(pred, duplicate)

    def test_blind_transform_has_no_outcome_argument(self):
        api = self.api()
        from prediction_extension.external import transform_blind
        state = {"features": ["s__a", "s__b"], "pseudocount": .5}
        raw = pd.DataFrame([[1.]], index=["external"], columns=["s__a"])
        with self.assertRaises(ValueError):
            transform_blind(raw, state)
        values, missing = transform_blind(raw, state, same_source_verified=True)
        self.assertEqual(missing, ["s__b"])
        np.testing.assert_allclose(values, [[np.log(3)/2, -np.log(3)/2]])
        with self.assertRaises(TypeError):
            transform_blind(raw, state, y=[1])
        duplicate = pd.DataFrame([[1., 2.]], columns=["s__a", "s__a"])
        with self.assertRaises(ValueError):
            transform_blind(duplicate, state, same_source_verified=True)


if __name__ == "__main__":
    unittest.main()
