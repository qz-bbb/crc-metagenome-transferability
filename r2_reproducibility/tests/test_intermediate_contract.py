import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd

class IntermediateContractTests(unittest.TestCase):
    def test_explicit_generated_dependency_replaces_old_path_with_same_hash_gate(self):
        mod=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);new=root/'new.tsv';new.write_text('recovered');contract=root/'contract.tsv'
            key='identity_multicohort_root/05_review/original_enriched_identity_manifest.tsv'
            contract.write_text('input_key\tsha256\n'+key+'\t'+mod.sha(new)+'\n')
            config={'identity_multicohort_root':root/'old_not_present'}
            self.assertEqual(mod.verify_contract(config,['identity_multicohort_root'],contract,overrides={key:new}),[key])
            new.write_text('tampered')
            with self.assertRaises(ValueError):
                mod.verify_contract(config,['identity_multicohort_root'],contract,overrides={key:new})

    def module(self):
        path=Path(__file__).resolve().parents[1]/'portable.py'
        spec=importlib.util.spec_from_file_location('portable_contract_test',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

    def test_legacy_matrices_accept_only_exact_canonical_values(self):
        mod=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'a.tsv';contract=root/'contract.tsv'
            frame=pd.DataFrame([[0.,1.234]],index=pd.Index(['s'],name='sample_uid'),columns=['a','b'])
            frame.to_csv(path,sep='\t')
            expected=mod.canonical_matrix_sha(path)
            for key in ['legacy_clr','legacy_abundance']:
                contract.write_text('input_key\tsha256\tcanonical_sha256\n'+key+'\t'+'0'*64+'\t'+expected+'\n')
                self.assertEqual(mod.verify_contract({key:path},[key],contract),[key])
                changed=frame.copy();changed.iloc[0,1]+=1e-12;changed.to_csv(path,sep='\t')
                with self.assertRaises(ValueError):mod.verify_contract({key:path},[key],contract)
                frame.to_csv(path,sep='\t')

    def test_canonical_gate_rejects_changed_axis(self):
        mod=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'a.tsv';contract=root/'contract.tsv'
            frame=pd.DataFrame([[1.,2.]],index=['s'],columns=['a','b']);frame.to_csv(path,sep='\t')
            expected=mod.canonical_matrix_sha(path)
            contract.write_text('input_key\tsha256\tcanonical_sha256\nlegacy_clr\t'+'0'*64+'\t'+expected+'\n')
            frame.columns=['b','a'];frame.to_csv(path,sep='\t')
            with self.assertRaises(ValueError):mod.verify_contract({'legacy_clr':path},['legacy_clr'],contract)

if __name__=='__main__':unittest.main()
