"""Behavioral tests for the portable boundary, written before its implementation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

RUNNER = Path(__file__).resolve().parents[1] / 'portable.py'


class PortableBoundaryTests(unittest.TestCase):
    def runner(self):
        self.assertTrue(RUNNER.is_file(), 'portable replay runner has not been implemented')
        spec = importlib.util.spec_from_file_location('portable', RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_relative_paths_resolve_against_config_not_cwd(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text(json.dumps({'source_root': 'inputs/source', 'output_root': 'results'}))
            got = module.load_config(config)
            self.assertEqual(got['source_root'], root / 'inputs/source')
            self.assertEqual(got['output_root'], root / 'results')

    def test_input_tree_is_not_accepted_as_output(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            with self.assertRaises(ValueError):
                module.check_output({'source_root': source, 'output_root': source / 'unsafe'})

    def test_existing_unmarked_directory_is_not_claimed(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'output').mkdir()
            (root / 'output' / 'unrelated.txt').write_text('preserve me')
            with self.assertRaises(ValueError):
                module.check_output({'source_root': root / 'source', 'output_root': root / 'output'})
            self.assertEqual((root / 'output' / 'unrelated.txt').read_text(), 'preserve me')

    def test_missing_reference_is_packaging_error_not_silent_fallback(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                module.check_code(Path(tmp))

    def test_missing_input_report_does_not_start_analysis(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = module.missing_inputs({'baseline_metadata': root / 'absent.tsv'}, ['baseline_metadata'])
            self.assertEqual(missing, ['baseline_metadata'])
            self.assertFalse((root / 'absent.tsv').exists())

    def test_input_hash_mismatch_fails_closed(self):
        module = self.runner()
        self.assertTrue(hasattr(module, 'verify_contract'), 'input hash gate is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.tsv'
            source.write_text('modified input')
            contract = root / 'contract.tsv'
            contract.write_text('input_key\tsha256\nbaseline_metadata\t' + '0' * 64 + '\n')
            with self.assertRaises(ValueError):
                module.verify_contract({'baseline_metadata': source}, ['baseline_metadata'], contract)

    def test_dynamic_record_metadata_does_not_require_identical_view_counts(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'zenodo_15069069.json').write_text('{"stats": {"views": 123}}')
            contract = root / 'contract.tsv'
            contract.write_text('input_key\tsha256\thash_policy\nsource_root/zenodo_15069069.json\t' +
                                '0' * 64 + '\tarchive_checksums_verified_by_audit\n')
            try:
                checked = module.verify_contract({'source_root': root}, ['source_root'], contract)
            except ValueError:
                self.fail('Dynamic record metadata was incorrectly treated as immutable data')
            self.assertEqual(checked, ['source_root/zenodo_15069069.json'])

    def test_exact_size_entry_validates_inputs_before_analysis(self):
        self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text(json.dumps({'output_root': str(root / 'new_results')}))
            result = subprocess.run([sys.executable, '-B', str(RUNNER), '--config', str(config),
                                     'nested-matched', '--smoke'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing required inputs', result.stderr)
            self.assertFalse((root / 'new_results').exists())


if __name__ == '__main__':
    unittest.main()

