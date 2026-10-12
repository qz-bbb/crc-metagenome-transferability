import importlib.util
from pathlib import Path
import tempfile,json,sys,unittest
sys.path.insert(0,str(Path(__file__).parents[1]))
from export_tables import export_groups

class TableExport(unittest.TestCase):
    def test_lossless_strings_and_relative_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); inp=root/'source';inp.mkdir()
            (inp/'a.tsv').write_text('taxon\tci975_low\nT\t-0.00002164397240382981\n')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'groups':[{'name':'Prediction_C_contrasts','files':['source/a.tsv']}]}))
            export_groups(manifest,root/'out')
            self.assertEqual((root/'out/Prediction_C_contrasts.tsv').read_text(),(inp/'a.tsv').read_text())
            self.assertIn('source/a.tsv',(root/'out/Prediction_sources.tsv').read_text())
    def test_duplicate_sheet_and_schema_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'input';inp.mkdir()
            (inp/'a.tsv').write_text('a\n1\n');(inp/'b.tsv').write_text('b\n2\n')
            for groups in [[{'name':'X','files':['input/a.tsv','input/b.tsv']}],
                [{'name':'X','files':['input/a.tsv']},{'name':'X','files':['input/a.tsv']}]]:
                manifest=root/'manifest.json';manifest.write_text(json.dumps({'groups':groups}))
                with self.assertRaises(ValueError):export_groups(manifest,root/'out')

if __name__=='__main__':unittest.main()
