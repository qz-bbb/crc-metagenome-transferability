"""Bounded official source acquisition and historical processed-input reconstruction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler

PACKAGE=Path(__file__).resolve().parent
HOSTS={'experimenthub.bioconductor.org','raw.githubusercontent.com','zenodo.org','mghp.osn.xsede.org'}
META_COLUMNS='sample_uid study_name sample_id subject_id body_site study_condition disease age age_category gender country location PMID NCBI_accession number_reads number_bases BMI disease_subtype disease_stage disease_location tnm ajcc fobt smoker ever_smoker alcohol alcohol_numeric sequencing_platform DNA_extraction_kit antibiotics_current_use non_westernized minimum_read_length median_read_length curator diet hdl ldl fasting_glucose creatinine triglycerides hba1c cholesterol brinkman_index ALT eGFR'.split()


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def verify_file(path,expected):
    if digest(path)!=expected: raise ValueError('Source hash mismatch: '+str(path))


def validate_url(url):
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ValueError('Unlisted or unsafe source URL')


def validate_member(member):
    name=PurePosixPath(member.name)
    if name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name or not (member.isfile() or member.isdir()):
        raise ValueError('Unsafe tar member: '+member.name)


def download(source,path,max_bytes=20_000_000):
    validate_url(source['url'])
    if source['bytes']>max_bytes: raise ValueError('Requested source exceeds byte ceiling')
    path=Path(path)
    if path.exists(): verify_file(path,source['sha256']); return {'status':'reused_verified','bytes':path.stat().st_size}
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial')
    if partial.exists(): raise ValueError('Existing partial file; inspect before retry')
    error=None
    # Direct unauthenticated HTTPS; never consult cookies, credentials or proxy config.
    opener=build_opener(ProxyHandler({}))
    for attempt in range(2):
        try:
            request=Request(source['url'],headers={'User-Agent':'CRC-reproducibility-source-check/1'})
            with opener.open(request,timeout=20) as response:
                validate_url(response.url)
                size=0
                with partial.open('xb') as out:
                    while True:
                        block=response.read(65536)
                        if not block: break
                        size+=len(block)
                        if size>max_bytes or size>source['bytes']: raise ValueError('Response exceeds frozen source size')
                        out.write(block)
            if size!=source['bytes']: raise ValueError('Source size mismatch')
            verify_file(partial,source['sha256'])
            if source.get('md5') and digest(partial,'md5')!=source['md5']: raise ValueError('Source MD5 mismatch')
            partial.rename(path)
            return {'status':'downloaded_verified','bytes':size,'attempts':attempt+1,'sha256':source['sha256']}
        except Exception as exc:
            if partial.exists(): partial.unlink()
            error=str(exc)
            if attempt==0: time.sleep(.5)
    raise RuntimeError('Bounded official source request failed: '+error)


def extract_mp4(root,inventory):
    root=Path(root); checked=[]
    expected=list(csv.DictReader((PACKAGE/'input_contract.tsv').open(encoding='utf8'),delimiter='\t'))
    for source in inventory:
        if source['family']!='mp4': continue
        archive=root/source['file'];verify_file(archive,source['sha256'])
        folder=source['archive_folder']; destination=root/'mp4_source/extracted'/folder
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers()
            for item in members: validate_member(item)
            names={Path(item.name).name:item for item in members if item.isfile()}
            if len(names)!=sum(item.isfile() for item in members): raise ValueError('Duplicate archive basenames')
            for row in expected:
                prefix='source_root/extracted/'+folder+'/'
                if not row['input_key'].startswith(prefix): continue
                basename=row['input_key'][len(prefix):]
                member=names[basename]; target=destination/basename
                payload=tar.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest()!=row['sha256']: raise ValueError('Extracted frozen file mismatch')
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists(): verify_file(target,row['sha256'])
                else: target.write_bytes(payload)
                checked.append(row['input_key'])
    # The established audit consumes this conventional name. Reconstruct a
    # clearly labeled checksum snapshot from the recorded official record;
    # mutable download/view statistics are never an exact hash requirement.
    snapshot={'id':15069069,'_kind':'verified checksum snapshot, not full live record JSON',
              '_source':'https://zenodo.org/api/records/15069069','files':[]}
    for source in inventory:
        if source['family']=='mp4':
            key='metadata.tar.gz' if source['archive_folder']=='metadata' else 'metaphlan4_profiles.tar.gz'
            snapshot['files'].append({'key':key,'size':source['bytes'],'checksum':'md5:'+source['md5'],
                                      'links':{'self':source['url']}})
    path=root/'mp4_source/zenodo_15069069.json';payload=json.dumps(snapshot,indent=2)+'\n'
    if path.exists() and path.read_text()!=payload:raise ValueError('Existing checksum record differs; preserve and inspect it')
    if not path.exists():path.write_text(payload)
    return checked


def reconstruct_cmd(root,inventory):
    import numpy as np
    import pandas as pd
    import pyreadr
    root=Path(root)
    sources=[row for row in inventory if row['family']=='cmd']
    for row in sources: verify_file(root/row['file'],row['sha256'])
    meta=pyreadr.read_r(str(root/'cmd/sampleMetadata.rda'))['sampleMetadata']
    studies=[row['study'] for row in sources if 'study' in row]
    # Historical ordering follows inventory studies and native source metadata
    # row order within each study, rather than lexically sorting sample IDs.
    selected=pd.concat([meta[meta.study_name.eq(study)&meta.study_condition.isin(['CRC','control'])]
                        for study in studies],ignore_index=True).copy()
    selected.insert(0,'sample_uid',selected.study_name.astype(str)+'__'+selected.sample_id.astype(str))
    selected=selected.reindex(columns=META_COLUMNS)
    if len(selected)!=1395 or int(selected.study_condition.eq('CRC').sum())!=701: raise ValueError('Historical cMD sample counts changed')
    blocks=[]
    for row in sources:
        if 'study' not in row: continue
        data=next(iter(pyreadr.read_r(str(root/row['file'])).values()))
        taxa=[name for name in data.index if '|s__' in name and '|t__' not in name]
        wanted=selected[selected.study_name.eq(row['study'])]
        block=data.loc[taxa,wanted.sample_id.tolist()].copy();block.columns=wanted.sample_uid.tolist()
        blocks.append(block)
    universe=sorted(set().union(*(set(block.index) for block in blocks)))
    matrix=pd.concat([block.reindex(universe,fill_value=0.) for block in blocks],axis=1).loc[:,selected.sample_uid]
    matrix.index.name='taxon_long'
    if matrix.shape!=(944,1395) or not np.isfinite(matrix.to_numpy()).all(): raise ValueError('Historical matrix shape/values invalid')
    destination=root/'cmd/reconstructed'
    if destination.exists(): raise ValueError('Reconstruction destination already exists')
    destination.mkdir(parents=True)
    selected.to_csv(destination/'curated_crc_case_control_sample_metadata.tsv',sep='\t',index=False)
    matrix.to_csv(destination/'curated_crc_case_control_species_relative_abundance.tsv.gz',sep='\t')
    return destination


def compare_cmd(reconstructed,frozen):
    import numpy as np
    import pandas as pd
    reconstructed=Path(reconstructed);frozen=Path(frozen)
    name='curated_crc_case_control_sample_metadata.tsv'
    a=pd.read_csv(reconstructed/name,sep='\t',keep_default_na=False)
    b=pd.read_csv(frozen/name,sep='\t',keep_default_na=False)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
    name='curated_crc_case_control_species_relative_abundance.tsv.gz'
    a=pd.read_csv(reconstructed/name,sep='\t',index_col=0,float_precision='round_trip')
    b=pd.read_csv(frozen/name,sep='\t',index_col=0,float_precision='round_trip')
    if list(a.index)!=list(b.index) or list(a.columns)!=list(b.columns): raise ValueError('Historical matrix identifiers/order changed')
    np.testing.assert_allclose(a,b,rtol=0,atol=0)
    return {'metadata_cells_identical':True,'matrix_values_identical':True,'matrix_shape':list(a.shape),
            'gzip_byte_identity_required':False,'max_absolute_difference':float(np.max(np.abs(a.to_numpy()-b.to_numpy())))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['download','extract-mp4','reconstruct-cmd','compare-cmd'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--family',choices=['cmd','mp4','all'],default='cmd')
    parser.add_argument('--frozen-cmd',type=Path);args=parser.parse_args()
    root=args.root.resolve()
    if root==PACKAGE or PACKAGE in root.parents: raise ValueError('Source work must be outside code package')
    inventory=json.loads((PACKAGE/'source_inventory.json').read_text())['sources']
    if args.action=='download':
        chosen=[s for s in inventory if args.family=='all' or s['family']==args.family]
        if sum(s['bytes'] for s in chosen)>20_000_000: raise ValueError('Batch exceeds 20 MB ceiling')
        result=[]
        for source in chosen:
            outcome=download(source,root/source['file']);outcome['file']=source['file'];result.append(outcome)
            print(source['file'],outcome['status'],flush=True)
    elif args.action=='extract-mp4': result={'verified_extracted_inputs':extract_mp4(root,inventory)}
    elif args.action=='reconstruct-cmd': result={'destination':str(reconstruct_cmd(root,inventory))}
    else:
        if not args.frozen_cmd: parser.error('--frozen-cmd required')
        result=compare_cmd(root/'cmd/reconstructed',args.frozen_cmd)
    root.mkdir(parents=True,exist_ok=True)
    (root/(args.action+'_receipt.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
