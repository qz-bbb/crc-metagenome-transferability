import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('report_paths',Path(__file__).parents[1]/'paths.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

class OutputBoundaries(unittest.TestCase):
    def test_fresh_sibling(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'input'; source.mkdir()
            target=Path(d)/'output'
            self.assertEqual(M.output_directory(target,[source]),target.resolve())
            self.assertTrue(target.is_dir())
    def test_input_overlap_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'input'; source.mkdir()
            for target in [source,source/'derived',Path(d)]:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    M.output_directory(target,[source])
            target=Path(d)/'existing'; target.mkdir(); (target/'receipt.txt').write_text('preserve')
            with self.assertRaises(ValueError):M.output_directory(target,[source])
            self.assertEqual((target/'receipt.txt').read_text(),'preserve')
    def test_code_directory_rejected(self):
        with self.assertRaises(ValueError):M.output_directory(Path(__file__).parents[1]/'generated',[])

if __name__=='__main__':unittest.main()
