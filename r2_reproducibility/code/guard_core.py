"""Independent provenance guards; never alter the original sensitivity lock."""
import hashlib
from pathlib import Path
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def verify_hashes(files):
    for path,expected in files.items():
        if sha(path)!=expected: raise ValueError('Frozen dependency changed: '+str(path))


def verify_selection(audit,selected,expected_n=1214):
    expected=audit[audit.accession_resolved].reset_index(drop=True)
    if len(expected)!=expected_n or len(selected)!=expected_n:
        raise ValueError('Unexpected selected training count')
    pd.testing.assert_frame_equal(expected,selected.reset_index(drop=True),check_dtype=False)

