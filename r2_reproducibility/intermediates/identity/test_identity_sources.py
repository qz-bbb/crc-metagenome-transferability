import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from identity_sources import build_enriched, validate_source, validate_output, resolve_audit, validate_ena_fields

HERE = Path(__file__).resolve().parent


class IdentitySourceTests(unittest.TestCase):
    def test_actual_ena_response_header_order(self):
        self.assertTrue(validate_ena_fields(['run_accession','study_accession','secondary_study_accession',
            'sample_accession','secondary_sample_accession','sample_alias','sample_title','experiment_alias','library_name']))

    def test_missing_ena_field_rejected(self):
        with self.assertRaisesRegex(ValueError,'field'):
            validate_ena_fields(['run_accession'])

    def original(self, uid='a', study='ThomasAM_2018a', alias='VF-114'):
        return {'sample_uid':uid, 'study_name':study, 'sample_id':uid,
                'subject_id':alias, 'NCBI_accession':''}

    def public_row(self, alias='vf114', run='SRR10'):
        return {'run_accession':run, 'sample_accession':'SAMN10',
                'secondary_sample_accession':'SRS10', 'sample_alias':alias,
                'sample_title':'', 'experiment_alias':'', 'library_name':''}

    def test_normalized_join_preserves_two_run_rows_and_source_order(self):
        rows=build_enriched([self.original()], {'PRJNA447983':[self.public_row(run='SRR20'),self.public_row()] ,'PRJEB27928':[]})
        self.assertEqual(rows,[{'sample_uid':'a','study_name':'ThomasAM_2018a','run_accession':r,
            'sample_accession':'SAMN10','secondary_sample_accession':'SRS10','sample_alias':'vf114',
            'match_basis':'same primary-source project; exact normalized supplied sample or subject alias'} for r in ['SRR20','SRR10']])

    def test_project_scope_blocks_cross_study_alias(self):
        rows=build_enriched([self.original(),self.original('w','WirbelJ_2018')],
                           {'PRJNA447983':[self.public_row()], 'PRJEB27928':[]})
        self.assertEqual([r['sample_uid'] for r in rows],['a'])

    def test_ambiguous_original_alias_rejected(self):
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            build_enriched([self.original(),self.original('b')],{'PRJNA447983':[self.public_row()],'PRJEB27928':[]})

    def test_duplicate_original_uid_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            build_enriched([self.original(),self.original()],{'PRJNA447983':[],'PRJEB27928':[]})

    def test_duplicate_public_run_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            build_enriched([self.original()],{'PRJNA447983':[self.public_row(),self.public_row()],'PRJEB27928':[]})

    def test_both_projects_required(self):
        with self.assertRaisesRegex(ValueError,'projects'):
            build_enriched([self.original()],{'PRJNA447983':[]})

    def test_generic_missing_tokens_do_not_join(self):
        self.assertEqual(build_enriched([self.original(alias='unspecified')],
            {'PRJNA447983':[self.public_row('UNSPECIFIED')],'PRJEB27928':[]}),[])

    def test_title_accession_can_join_without_fuzzy_alias(self):
        source=self.original(); source['NCBI_accession']='SRR10'
        public=self.public_row('unrelated',run='SRR99');public['sample_title']='metadata SRR10 sample'
        self.assertEqual(len(build_enriched([source],{'PRJNA447983':[public],'PRJEB27928':[]})),1)

    def files(self, root):
        payload=b'run_accession\nSRR1\n'; path=root/'snapshot.tsv';path.write_bytes(payload)
        expected={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'url':'https://www.ebi.ac.uk/ena/test'}
        prov=root/'snapshot.provenance.json';prov.write_text(json.dumps({**expected,'retrieved_utc':'2026-10-10T00:00:00+00:00'}),encoding='utf8')
        return path,prov,expected

    def test_valid_source_provenance_retained(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path,prov,expected=self.files(Path(folder))
            self.assertEqual(validate_source(path,prov,expected).get('retrieved_utc'),'2026-10-10T00:00:00+00:00')

    def test_tampered_payload_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path,prov,expected=self.files(Path(folder));path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Source'):
                validate_source(path,prov,expected)

    def test_wrong_provenance_url_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path,prov,expected=self.files(Path(folder));data=json.loads(prov.read_text());data['url']='https://wrong.example/'
            prov.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'provenance'):
                validate_source(path,prov,expected)

    def test_output_overlap_and_existing_directory_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'overlap'):validate_output(root/'out',[root])
            with self.assertRaisesRegex(ValueError,'exists'):validate_output(root,[])

    def test_resolution_priority_full_token_filter_and_union(self):
        training=pd.DataFrame({'sample_uid':['a','b','c','d'],'sample_id':['A','SAMD12','C','D'],
                              'NCBI_accession':['SRR1','', '', 'alias-SRR3']})
        enriched=pd.DataFrame({'sample_uid':['c','c'],'run_accession':['SRR9','SRR8'],
                               'sample_accession':['SAMN9',''],'secondary_sample_accession':['','SRS8']})
        out=resolve_audit(training,enriched)
        self.assertEqual(list(out.columns),list(training.columns)+['accession_resolved','resolution_basis','resolved_accessions'])
        self.assertEqual(out.resolution_basis.tolist(),['direct','biosample_id','enriched','alias_only'])
        self.assertEqual(out.resolved_accessions.tolist(),['SRR1','SAMD12','SAMN9;SRR8;SRR9;SRS8',''])

    def test_duplicate_training_identity_rejected(self):
        training=pd.DataFrame({'sample_uid':['a','a'],'sample_id':['A','B'],'NCBI_accession':['','']})
        enriched=pd.DataFrame(columns=['sample_uid','run_accession'])
        with self.assertRaisesRegex(ValueError,'unique'):resolve_audit(training,enriched)


if __name__=='__main__':unittest.main()
