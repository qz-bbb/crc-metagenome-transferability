import unittest
import numpy as np
import pandas as pd
from ablation_core import eligibility, panels, abundance_summary, bootstrap_metrics

class Tests(unittest.TestCase):
    def evidence(self):
        return pd.DataFrame(dict(taxon=['b','a','c','d','e','f'],q=[.01,.01,.03,.10,.02,.04],
            direction_agreement=[.6,.59,.9,1.,.6,.6],re_q=[.09,.01,.10,.01,.01,.01],
            i2=[74.,1.,1.,1.,75.,1.],inner_sign_agreement=[.9,.9,1.,1.,1.,.89],
            inner_q_fraction=[.8,.8,1.,1.,1.,.8]))
    def test_boundaries(self):
        r=eligibility(self.evidence()); self.assertEqual(r.A.sum(),5)
        self.assertEqual(r.loc[r.ICP,'taxon'].tolist(),['b'])
    def test_ranking_no_padding(self):
        p,n=panels(self.evidence()); self.assertEqual(n,1); self.assertEqual(p['A'],['a'])
        self.assertEqual(p['ICP'],['b']); self.assertTrue(all(len(v)==1 for v in p.values()))
    def test_empty(self):
        a=self.evidence(); a['inner_q_fraction']=0; p,n=panels(a)
        self.assertEqual(n,0); self.assertTrue(all(not v for v in p.values()))
    def test_zero_and_positive(self):
        x=np.array([[0.,1.],[0.,3.],[1.,0.],[0.,0.]])
        r=abundance_summary(x,np.array([0,0,1,1]),['a','b'])
        self.assertEqual(r.iloc[0].positive_mean_crc,1.)
        self.assertTrue(np.isnan(r.iloc[0].positive_mean_control))
        self.assertEqual(r.iloc[1].positive_mean_control,2.)
    def test_ties(self):
        y=np.array([0,0,1,1]); p={'A':np.full(4,.5),'ICP':np.full(4,.5)}
        point,boot=bootstrap_metrics(y,p,20,1)
        np.testing.assert_allclose(point['A'],[.5,.5]); np.testing.assert_allclose(boot['ICP']-boot['A'],0)
    def test_selection_no_labels_interface(self):
        import inspect
        self.assertEqual(list(inspect.signature(panels).parameters),['stats'])

if __name__=='__main__': unittest.main()

