"""Lossless TSV counterparts of the new S4 sheets; no refits or recalculation."""
from pathlib import Path
import argparse,csv,hashlib,json
from paths import output_directory

def export_groups(manifest_path,output):
    manifest_path=Path(manifest_path).resolve()
    config=json.loads(manifest_path.read_text(encoding='utf-8'))
    groups=config['groups']; names=[g['name'] for g in groups]
    if not groups or len(names)!=len(set(names)) or any(not n or Path(n).name!=n or n in ['Prediction_notes','Prediction_sources'] for n in names):
        raise ValueError('Unique safe sheet names required')
    records=[]; sources=[]; protected=[manifest_path]
    for group in groups:
        if not group['files']:raise ValueError('Missing source file list')
        headers=None; rows=[]
        for entry in group['files']:
            entry={'path':entry} if isinstance(entry,str) else entry
            relative=entry['path'];path=(manifest_path.parent/relative).resolve()
            protected.append(path)
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f,delimiter='\t');current=reader.fieldnames;part=list(reader)
            if not current or len(current)!=len(set(current)) or any(None in row or any(v is None for v in row.values()) for row in part):
                raise ValueError('Invalid source TSV schema')
            if 'held_out_study' in entry and 'held_out_study' not in current:
                current=['held_out_study',*current]
                part=[{'held_out_study':entry['held_out_study'],**r} for r in part]
            if headers is not None and current!=headers:raise ValueError('Cannot concatenate unlike schemas')
            headers=current;rows.extend(part)
            sources.append([group['name'],relative,hashlib.sha256(path.read_bytes()).hexdigest(),len(part)])
        records.append((group['name'],headers,rows))
    destination=output_directory(output,protected)
    for name,headers,rows in records:
        with (destination/(name+'.tsv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=headers,delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (destination/'Prediction_sources.tsv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['sheet','relative_file','sha256','rows']);writer.writerows(sources)
    notes=[['scope','TSV counterparts of S4 Prediction sheets, not a reconstruction of historical workbook formatting.'],
           ['timing','Post-outcome sensitivity; external cohorts were previously examined.'],
           ['precision','Exact source strings retained, including negative near-zero confidence limits.'],
           ['uncertainty','Fixed-model conditional participant bootstrap; not development or new-population uncertainty.'],
           ['privacy','Locally generated probability sheets contain individual sample IDs; do not upload them automatically.']]
    with (destination/'Prediction_notes.tsv').open('w',encoding='utf-8',newline='') as f:
        csv.writer(f,delimiter='\t',lineterminator='\n').writerows([['item','description'],*notes])
    return {'groups':len(records),'data_rows':sum(len(r[2]) for r in records),'source_files':len(sources)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export_groups(args.manifest,args.output)))
