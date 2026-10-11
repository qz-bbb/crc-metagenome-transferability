"""New cohort gates and paired candidate-replication endpoints."""
import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
ZERO_TOL=1e-12  # Numerical CLR cancellation tolerance, not a biological cutoff.

def direction_fraction(effect, reference):
    a,b=np.asarray(effect),np.asarray(reference)
    if a.shape!=b.shape or a.size==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid effect vectors')
    return float(np.mean((np.sign(a)==np.sign(b)) & (np.abs(a)>ZERO_TOL) & (np.abs(b)>ZERO_TOL)))

def bootstrap_effects(x,y,n_boot,seed):
    x=np.asarray(x,dtype=float); y=np.asarray(y)
    if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Finite matrix and both binary groups required')
    x=x-x[0]
    rng=np.random.default_rng(seed); means=[]
    for value in (0,1):
        a=x[y==value]; n=len(a)
        weights=rng.multinomial(n,np.full(n,1/n),size=n_boot)
        means.append(weights@a/n)
    effects=means[1]-means[0]
    effects[np.abs(effects)<=ZERO_TOL]=0.
    return effects

def bootstrap_directions(x,y,raw,reference,n_boot,seed):
    x=np.asarray(x,dtype=float); y=np.asarray(y); raw=np.asarray(raw,dtype=float); reference=np.asarray(reference)
    if raw.shape!=x.shape or not np.isfinite(raw).all() or (raw<0).any() or reference.shape!=(x.shape[1],):
        raise ValueError('Invalid detection matrix/reference')
    effects=bootstrap_effects(x,y,n_boot,seed)
    rng=np.random.default_rng(seed); detected=np.zeros(effects.shape,dtype=bool)
    for value in (0,1):
        present=(raw[y==value]>0).astype(int); n=len(present)
        weights=rng.multinomial(n,np.full(n,1/n),size=n_boot)
        detected |= (weights@present)>0
    return detected & (np.sign(effects)==np.sign(reference)[None,:]) & (np.abs(effects)>ZERO_TOL) & (np.abs(reference)[None,:]>ZERO_TOL)

def welch_effects(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y)
    if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Finite matrix and both binary groups required')
    x=x-x[0]
    a,b=x[y==1],x[y==0]
    if min(len(a),len(b))<2: raise ValueError('At least two subjects per group')
    effect=a.mean(axis=0)-b.mean(axis=0)
    effect[np.abs(effect)<=ZERO_TOL]=0.
    va,vb=a.var(axis=0,ddof=1)/len(a),b.var(axis=0,ddof=1)/len(b)
    total=va+vb; p=np.ones(x.shape[1]); valid=total>0
    df=total[valid]**2/(va[valid]**2/(len(a)-1)+vb[valid]**2/(len(b)-1))
    p[valid]=2*t.sf(np.abs(effect[valid])/np.sqrt(total[valid]),df)
    p[(~valid)&(effect!=0)]=0.
    if not np.isfinite(p).all(): raise ValueError('Invalid Welch p values')
    return effect,p,multipletests(p,method='fdr_bh')[1]

def check_ids(meta,profile_ids,training_ids):
    ids=meta['Sample Source ID'].astype(str)
    if ids.duplicated().any() or len(set(profile_ids))!=len(profile_ids): raise ValueError('Duplicate specimen IDs')
    if set(ids)!=set(profile_ids): raise ValueError('Metadata/profile mismatch')
    if set(ids)&set(training_ids): raise ValueError('Training overlap')
    if ids.eq('').any(): raise ValueError('Empty specimen ID')
    return True

