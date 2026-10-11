import pandas as pd
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

def read_exact(path,**kwargs):
    kwargs.setdefault('sep','\t')
    kwargs.setdefault('float_precision','round_trip')
    return pd.read_csv(path,**kwargs)

def rebuild_aggregate_exact(sources,output,receipt_dir):
    """Rebuild only a derived table, on every call, with append-only repair logs."""
    sources=[Path(p) for p in sources];output=Path(output);receipt_dir=Path(receipt_dir)
    if output.resolve() in {p.resolve() for p in sources}:
        raise ValueError('Aggregate must not overwrite a source')
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    frame=pd.concat([read_exact(p) for p in sources],ignore_index=True)
    expected=frame.to_csv(None,sep='\t',index=False,float_format='%.17g').encode('utf8')
    before=output.read_bytes() if output.exists() else b''
    changed=before!=expected
    receipt={'changed':changed,'reason':'regenerate from immutable fold files using exact binary float round-trip',
             'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(expected).hexdigest(),
             'source_hashes':source_hashes,'thresholds_changed':False,'models_or_predictions_changed':False}
    if changed:
        output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(expected)
        receipt_dir.mkdir(parents=True,exist_ok=True)
        now=datetime.now(timezone.utc);receipt['utc']=now.isoformat()
        path=receipt_dir/(now.strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
        with path.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
    for p,h in source_hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise AssertionError('Source changed during aggregate repair')
    return receipt

