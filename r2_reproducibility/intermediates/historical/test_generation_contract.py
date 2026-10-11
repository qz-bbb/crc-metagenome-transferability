import json
from pathlib import Path
import tempfile
import unittest

class ContractTests(unittest.TestCase):
    def test_preflight_rejects_changed_source_before_output(self):
        import historical_intermediates as h
        self.assertTrue(hasattr(h,'verify_sources'),'Frozen-source preflight is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=root/'meta.tsv';m.write_text('changed');a=root/'matrix.tsv';a.write_text('wrong')
            config={'baseline_metadata':m,'baseline_matrix':a,'output_root':root/'out','targets':{}}
            with self.assertRaisesRegex(ValueError,'Frozen source mismatch'):
                h.generate(config,'matrices')
            self.assertFalse(config['output_root'].exists())
    def test_source_stability_gate_raises(self):
        import historical_intermediates as h
        self.assertTrue(hasattr(h,'verify_unchanged'),'Source-stability gate is missing')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source';p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed during'):
                h.verify_unchanged({str(p):'0'*64})

if __name__=='__main__':unittest.main()
