"""Fail-closed accession-resolution and paired-sensitivity boundaries."""
import re
import numpy as np

ACCESSION=re.compile(r'(?:[SED]RR\d+|[SED]RS\d+|SAM[NED][A-Z]*\d+)')

def accession_tokens(value):
    return {s for s in re.split(r'[;,\s]+',str(value).strip()) if ACCESSION.fullmatch(s)}

def resolve_training(training,enriched):
    if training.sample_uid.duplicated().any() or training.sample_id.duplicated().any():
        raise ValueError('Training sample identity is not unique')
    linked={}
    for row in enriched.to_dict('records'):
        acc=set()
        for field in ('run_accession','sample_accession','secondary_sample_accession'):
            acc |= accession_tokens(row.get(field,''))
        linked.setdefault(row['sample_uid'],set()).update(acc)
    out=training.copy(); resolved=[]; sources=[]; accessions=[]
    for row in out.to_dict('records'):
        direct=accession_tokens(row.get('NCBI_accession',''))
        sid=accession_tokens(row['sample_id'])
        added=linked.get(row['sample_uid'],set())
        acc=direct|sid|added
        resolved.append(bool(acc)); accessions.append(';'.join(sorted(acc)))
        sources.append('direct' if direct else 'biosample_id' if sid else 'enriched' if added else 'alias_only')
    out['accession_resolved']=resolved
    out['resolution_basis']=sources
    out['resolved_accessions']=accessions
    return out

def assert_disjoint_ids(training,external):
    if set(training)&set(external): raise ValueError('Training/external sample ID overlap')

def paired_delta(new,old):
    if new.shape!=old.shape: raise ValueError('Unpaired bootstrap arrays')
    return np.quantile(new-old,[.025,.975],axis=0)

