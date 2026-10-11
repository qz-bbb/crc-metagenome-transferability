import hashlib
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from guard_core import verify_hashes, verify_selection


class GuardTests(unittest.TestCase):
    def test_hash_drift_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.txt'; p.write_bytes(b'original')
            manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
            verify_hashes(manifest)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_hashes(manifest)

    def test_selection_order_and_values_fail(self):
        audit=pd.DataFrame({'sample_uid':['a','b','c'],'sample_id':['x','y','z'],
                            'accession_resolved':[True,False,True]})
        selected=audit.iloc[[0,2]].reset_index(drop=True)
        verify_selection(audit,selected,expected_n=2)
        with self.assertRaises(AssertionError):
            verify_selection(audit,selected.iloc[::-1].reset_index(drop=True),expected_n=2)
        changed=selected.copy(); changed.loc[0,'sample_id']='other'
        with self.assertRaises(AssertionError): verify_selection(audit,changed,expected_n=2)

if __name__=='__main__': unittest.main()

