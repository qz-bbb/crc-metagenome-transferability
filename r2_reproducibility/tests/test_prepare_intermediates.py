import importlib.util
from pathlib import Path
import tempfile
import unittest

class PrepareTests(unittest.TestCase):
    def module(self):
        p=Path(__file__).resolve().parents[1]/'prepare_intermediates.py'
        self.assertTrue(p.exists(),'Explicit intermediate entrypoint is missing')
        s=importlib.util.spec_from_file_location('prepare_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    def test_explicit_stage_only(self):
        m=self.module()
        with self.assertRaises(ValueError):m.command('all',Path('config.json'))
    def test_historical_stage_forwards_one_absolute_config(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'config.json';c.write_text('{}')
            args=m.command('historical-full',c)
            self.assertEqual(args[-3:],['--config',str(c.resolve()),'full'])
            self.assertIn('-B',args)

if __name__=='__main__':unittest.main()
