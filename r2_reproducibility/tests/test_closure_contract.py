"""Behavior contracts recorded before closure runner implementation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ClosureContracts(unittest.TestCase):
    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            p.write_text(json.dumps({'output_root': 'out', 'unexpected': 'x'}))
            with self.assertRaises(ValueError): module('closures').load_config(p)

    def test_relative_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            p.write_text(json.dumps({'output_root': 'out', 'baseline_metadata': 'inputs/m.tsv'}))
            self.assertEqual(module('closures').load_config(p)['baseline_metadata'], Path(tmp) / 'inputs/m.tsv')

    def test_nested_source_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                module('closures').check_output({'nested_reference_root': root / 'source', 'output_root': root / 'source/out'})

    def test_existing_unmarked_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): module('closures').check_output({'output_root': Path(tmp)})

    def test_missing_dependencies_create_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                module('closures').begin({'output_root': root / 'new'}, ['baseline_metadata'])
            self.assertFalse((root / 'new').exists())

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'x'; p.write_text('altered')
            with self.assertRaises(ValueError): module('source_access').verify_file(p, '0' * 64)

    def test_archive_traversal_and_links_rejected(self):
        import tarfile
        access = module('source_access')
        for name, kind in [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE), ('safe/link', tarfile.SYMTYPE)]:
            item = tarfile.TarInfo(name); item.type = kind
            with self.assertRaises(ValueError): access.validate_member(item)

    def test_http_and_unlisted_host_rejected(self):
        access = module('source_access')
        for url in ['http://experimenthub.bioconductor.org/fetch/1', 'https://example.org/data']:
            with self.assertRaises(ValueError): access.validate_url(url)

    def test_byte_ceiling_enforced_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                module('source_access').download({'url': 'https://experimenthub.bioconductor.org/fetch/1', 'bytes': 1000, 'sha256': '0'*64}, Path(tmp)/'x', max_bytes=10)

    def test_canonical_matrix_accepts_serialization_not_value_changes(self):
        import gzip
        portable = module('portable')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = root/'a.tsv.gz'; b = root/'b.tsv.gz'
            a.write_bytes(gzip.compress(b'taxon_long\ts1\nspecies\t1.0\n',mtime=0))
            b.write_bytes(gzip.compress(b'taxon_long\ts1\nspecies\t1\n',mtime=99))
            self.assertEqual(portable.canonical_matrix_sha(a),portable.canonical_matrix_sha(b))
            b.write_bytes(gzip.compress(b'taxon_long\ts1\nspecies\t1.01\n',mtime=99))
            self.assertNotEqual(portable.canonical_matrix_sha(a),portable.canonical_matrix_sha(b))


if __name__ == '__main__': unittest.main()
