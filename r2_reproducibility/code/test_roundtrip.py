import unittest
from io import StringIO
import pandas as pd
from exact_tsv import read_exact

class RoundTrip(unittest.TestCase):
    def test_rule_threshold_survives_17_digit_tsv(self):
        s='direction_agreement\tinner_sign_agreement\tinner_q_fraction\n0.59999999999999998\t0.90000000000000002\t0.80000000000000004\n'
        r=read_exact(StringIO(s)).iloc[0]
        self.assertEqual(r.direction_agreement,.6)
        self.assertEqual(r.inner_sign_agreement,.9)
        self.assertEqual(r.inner_q_fraction,.8)

    def test_repeated_aggregate_repair_is_idempotent_and_keeps_sources(self):
        import tempfile
        from pathlib import Path
        from exact_tsv import rebuild_aggregate_exact
        with tempfile.TemporaryDirectory() as td:
            b=Path(td);source=b/'source.tsv';out=b/'aggregate.tsv';receipts=b/'repair_receipts'
            source.write_text('direction_agreement\n0.59999999999999998\n',encoding='utf8')
            original=source.read_bytes()
            for iteration in range(2):
                # Reproduce an ordinary finalizer's lossy parse/serialize cycle.
                pd.read_csv(source,sep='\t').to_csv(out,sep='\t',index=False,float_format='%.17g')
                self.assertLess(read_exact(out).iloc[0].direction_agreement,.6)
                self.assertTrue(rebuild_aggregate_exact([source],out,receipts)['changed'])
                self.assertEqual(read_exact(out).iloc[0].direction_agreement,.6)
                self.assertFalse(rebuild_aggregate_exact([source],out,receipts)['changed'])
                self.assertEqual(source.read_bytes(),original)
            self.assertEqual(len(list(receipts.glob('*.json'))),2)

if __name__=='__main__':unittest.main()

