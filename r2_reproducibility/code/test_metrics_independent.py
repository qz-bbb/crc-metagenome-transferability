"""Independent tied-score rank AUC / grouped-threshold AP cross-check."""
import unittest
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score

class MetricCrosscheck(unittest.TestCase):
    def test_100_tied_score_datasets(self):
        rng=np.random.default_rng(20261010)
        for _ in range(100):
            n=int(rng.integers(10,200));y=rng.integers(0,2,n)
            y[0]=0;y[1]=1;p=rng.integers(0,6,n)/5
            n1=y.sum();n0=n-n1
            auc=(rankdata(p,method='average')[y==1].sum()-n1*(n1+1)/2)/(n0*n1)
            ap=0.;prior_recall=0.
            for threshold in sorted(set(p),reverse=True):
                hit=p>=threshold;tp=y[hit].sum();recall=tp/n1
                ap+=(recall-prior_recall)*tp/hit.sum();prior_recall=recall
            self.assertAlmostEqual(auc,roc_auc_score(y,p),places=13)
            self.assertAlmostEqual(ap,average_precision_score(y,p),places=13)

if __name__=='__main__':unittest.main()

