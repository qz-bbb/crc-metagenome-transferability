"""Behavioral checks: numeric transforms, axis guards, source isolation, and writes."""
import importlib
import tempfile
import subprocess
import sys
import json
import hashlib
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    core = importlib.import_module('historical_core')
except ModuleNotFoundError:
    core = None


class HistoricalTests(unittest.TestCase):
    def fixture_runner(self, root, inputs):
        # A separate toy deployment has its own frozen fixture contract. The
        # shipped scientific contract is never altered or bypassed.
        code=root/'code';code.mkdir()
        for name in ['historical_intermediates.py','historical_core.py']:
            shutil.copyfile(Path(__file__).parent/name,code/name)
        source={'baseline_metadata':inputs/'metadata.tsv','baseline_matrix':inputs/'matrix.tsv'}
        contract={'inputs':{k:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in source.items()},'algorithms':{}}
        (code/'generation_contract.json').write_text(json.dumps(contract))
        return code/'historical_intermediates.py'

    def setUp(self):
        self.assertIsNotNone(core, 'historical generator API is missing')
        self.meta = pd.DataFrame({'sample_uid': ['b', 'a', 'c'],
                                  'study_name': ['S2', 'S1', 'S3'],
                                  'study_condition': ['CRC', 'control', 'CRC']})
        self.raw = pd.DataFrame([[1., 1., 3.], [2., 4., 8.], [0., 0., 0.]],
                                index=['z', 'a', 'absent'], columns=['b', 'a', 'c'])

    def test_filter_preserves_both_axes_and_percent_units(self):
        abundance, clr, pc = core.transform(self.raw, self.meta)
        self.assertEqual(list(abundance.columns), ['z', 'a'])
        self.assertEqual(list(abundance.index), ['b', 'a', 'c'])
        self.assertEqual(abundance.iloc[2].tolist(), [3., 8.])
        self.assertEqual(pc, 0.5)
        np.testing.assert_allclose(clr.iloc[0], [-0.2554128118829953, 0.2554128118829953], atol=1e-14, rtol=0)

    def test_filter_drops_taxon_absent_from_three_studies(self):
        self.raw.loc['z', 'a'] = 0
        self.raw.loc['z', 'c'] = 0
        abundance, _, _ = core.transform(self.raw, self.meta)
        self.assertEqual(list(abundance.columns), ['a'])

    def test_order_mismatch_rejected_instead_of_reordered(self):
        with self.assertRaisesRegex(ValueError, 'order'):
            core.transform(self.raw[['a', 'b', 'c']], self.meta)

    def test_duplicate_axes_and_bad_values_rejected(self):
        for mutation in ['duplicate', 'infinite', 'negative', 'bad_label']:
            raw, meta = self.raw.copy(), self.meta.copy()
            if mutation == 'duplicate': raw.index = ['z', 'z', 'absent']
            elif mutation == 'infinite': raw.iloc[0, 0] = np.inf
            elif mutation == 'negative': raw.iloc[0, 0] = -1
            else: meta.loc[0, 'study_condition'] = 'adenoma'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                core.transform(raw, meta)

    def test_output_ancestor_and_existing_directory_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'sources' / 'matrix.tsv'
            source.parent.mkdir()
            source.write_text('input')
            for output in [root, source.parent, source.parent / 'output']:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    core.fresh_output(output, [source])
            target = root / 'fresh'
            core.fresh_output(target, [source])
            with self.assertRaises(ValueError): core.fresh_output(target, [source])

    def test_function_loader_omits_top_level_and_main_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'algorithm.py'
            source.write_text('from pathlib import Path\nPath("TOP_LEVEL_DANGER").write_text("bad")\ndef add(a,b):\n return a+b\ndef main():\n Path("MAIN_DANGER").write_text("bad")\nmain()\n')
            module = core.safe_functions(source, ['add'])
            self.assertEqual(module['add'](3, 4), 7)
            self.assertNotIn('main', module)

    def test_numeric_comparison_does_not_erase_missingness_or_order(self):
        left = pd.DataFrame({'taxon': ['b', 'a'], 'v': [1., np.nan]})
        right = left.copy()
        self.assertTrue(core.compare_frames(left, right, 1e-12)['pass'])
        right.loc[1, 'v'] = 0
        self.assertFalse(core.compare_frames(left, right, 1e-12)['pass'])
        self.assertFalse(core.compare_frames(left, left.iloc[::-1].reset_index(drop=True), 1e-12)['pass'])

    def test_fragment_executes_only_named_main_range_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'algorithm.py'
            source.write_text('import pandas as pd\ndef main():\n x=2\n result=pd.DataFrame({"n":[x+3]})\n result.to_csv("DANGER")\n end=result["n"].sum()\n raise RuntimeError("outside")\n')
            context = core.safe_functions(source, [])
            core.fragment(source, 'x=2', 'end=', context)
            self.assertEqual(context['end'], 5)

    def test_cli_generates_mapped_matrices_and_refuses_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'inputs'
            inputs.mkdir()
            self.raw.to_csv(inputs / 'matrix.tsv', sep='\t')
            self.meta.to_csv(inputs / 'metadata.tsv', sep='\t', index=False)
            config = root / 'config.json'
            config.write_text(json.dumps({'baseline_matrix': 'inputs/matrix.tsv',
                                           'baseline_metadata': 'inputs/metadata.tsv',
                                           'output_root': 'generated'}))
            cmd = [sys.executable, '-B', str(self.fixture_runner(root, inputs)),
                   '--config', str(config), 'matrices']
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            mapping = json.loads((root / 'generated' / 'output_mapping.json').read_text())
            result = core.read_matrix(root / 'generated' / mapping['legacy_abundance'])
            self.assertEqual(result.loc['c', 'a'], 8.)
            self.assertEqual(result.columns.tolist(), ['z', 'a'])
            repeat = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(result.loc['c', 'a'], 8.)

    def test_duplicate_raw_header_is_rejected_before_pandas_mangling(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'bad.tsv'
            source.write_text('taxon\tx\tx\na\t1\t2\n')
            with self.assertRaises(ValueError): core.read_matrix(source)

    def test_missing_comparison_target_does_not_erase_generation_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'inputs'
            inputs.mkdir()
            self.raw.to_csv(inputs / 'matrix.tsv', sep='\t')
            self.meta.to_csv(inputs / 'metadata.tsv', sep='\t', index=False)
            config = root / 'config.json'
            config.write_text(json.dumps({'baseline_matrix': 'inputs/matrix.tsv',
                    'baseline_metadata': 'inputs/metadata.tsv', 'output_root': 'generated',
                    'targets': {'legacy_clr': 'inputs/not_available.tsv.gz'}}))
            run = subprocess.run([sys.executable, '-B', str(self.fixture_runner(root, inputs)),
                                   '--config', str(config), 'matrices'], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            result = json.loads((root/'generated'/'comparisons.json').read_text())
            self.assertFalse(result['legacy_clr']['pass'])
            self.assertEqual(result['legacy_clr']['status'], 'comparison_target_missing')
            self.assertTrue((root/'generated'/'generation_receipt.json').is_file())

    def test_clean_snapshot_retains_algorithm_and_discards_private_setup(self):
        self.assertTrue(hasattr(core, 'snapshot_source'), 'clean source snapshot API missing')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.py'
            source.write_text('from pathlib import Path\nPRIVATE=Path("private/path")\ndef add(a,b):\n return a+b\ndef main():\n result=add(2,3)\n end=result+1\n raise RuntimeError("broad main")\nmain()\n')
            text = core.snapshot_source(source, ['add'], [('result=', 'end=')])
            target = Path(tmp) / 'clean.py'
            target.write_text(text)
            env = core.safe_functions(target, ['add'])
            core.fragment(target, 'result =', 'end =', env)
            self.assertEqual(env['end'], 6)
            self.assertNotIn('PRIVATE', env)

    def test_historical_checkpoint_replays_default_decimal_parser(self):
        self.assertTrue(hasattr(core, 'checkpoint'), 'historical TSV boundary missing')
        frame = pd.DataFrame({'n': [0.12345678901234568], 'label': ['preserved']}, index=['sample'])
        parsed = core.checkpoint(frame, index=True)
        self.assertEqual(parsed.iloc[0]['n'], 0.1234567890123456)
        self.assertEqual(parsed.index.tolist(), ['sample'])
        self.assertEqual(parsed.iloc[0]['label'], 'preserved')

    def test_blank_taxon_and_metadata_labels_rejected(self):
        for kind in ['taxon', 'study']:
            raw, meta = self.raw.copy(), self.meta.copy()
            if kind == 'taxon': raw.index = [' ', 'a', 'absent']
            else: meta.loc[0, 'study_name'] = ' '
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.transform(raw, meta)


if __name__ == '__main__':
    unittest.main(verbosity=2)
