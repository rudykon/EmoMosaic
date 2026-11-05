"""Tests mathematical invariants, not scores or duplicated implementation details."""
import unittest
import numpy as np
import torch
import yaml
from data import ROOT,zero_runs,raw_arrays
from missing import interval_mask,fixed_mask,training_mask
from model import RMNet


class Invariants(unittest.TestCase):
    def test_segments_and_padding(self):
        rng=np.random.default_rng(32)
        for L in range(1,51):
            for count in range(L+1):
                for segments in [1,2,4]:
                    z=interval_mask(L,count,segments,rng)
                    self.assertEqual(z.sum(),count)
                    self.assertFalse(z[L:].any())
                    self.assertEqual(len(zero_runs(z)),min(segments,count,L-count+1))
        lengths=np.array([3,7,25,50])
        mask=fixed_mask(lengths,'TAV',.4,201,colocated=True)
        self.assertTrue(np.array_equal(mask[:,0],mask[:,1]))
        self.assertTrue(np.array_equal(mask,fixed_mask(lengths,'TAV',.4,201,colocated=True)))
        for i,L in enumerate(lengths):self.assertFalse(mask[i,:,L:].any())

    def test_numeric_model_contract(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        cfg=yaml.safe_load((ROOT/'src/p2p3/config.yaml').read_text())
        model=RMNet(cfg).eval();lengths=torch.tensor([50,20,7,4])
        valid=torch.arange(50)[None]<lengths[:,None]
        obs=valid[:,None].expand(-1,3,-1).clone();obs[1,0]=False;obs[2]=False
        xs=[torch.randn(4,50,d) for d in cfg['dimensions']]
        with torch.no_grad():
            p=model(xs,obs,lengths)
            # A masked value must have no effect even if replaced with enormous numbers.
            altered=[torch.where(obs[:,m,:,None],x,torch.full_like(x,1e6)) for m,x in enumerate(xs)]
            q=model(altered,obs,lengths)
            self.assertTrue(torch.allclose(p['y'],q['y'],atol=1e-6))
            self.assertTrue(torch.isfinite(p['y']).all())
            self.assertTrue((p['y'].abs()<=3).all())
            self.assertTrue(torch.allclose(p['y'],p['contributions'].sum(1)+p['y_int'],atol=1e-6))
            self.assertEqual(float(p['w'][1,0]),0.)
            self.assertEqual(float(p['w'][2].sum()),0.)
            self.assertTrue(torch.equal(p['alpha'][~obs],torch.zeros_like(p['alpha'][~obs])))
            self.assertTrue(torch.allclose(p['alpha'].sum(-1),obs.any(-1).float(),atol=1e-6))
            self.assertTrue(torch.allclose(p['w'].sum(-1),obs.any(-1).any(-1).float(),atol=1e-6))
            full=model(xs,torch.ones_like(obs),lengths,mask=obs)
            self.assertTrue(torch.allclose(full['y'],p['y'],atol=1e-6))

    def test_raw_zero_detection(self):
        d={m:np.ones((1,50,k),np.float32) for m,k in [('text',768),('audio',74),('vision',35)]}
        t=np.zeros((1,3,50));t[0,0,:5]=[101,2000,0,2001,102];t[0,1,:5]=[1,1,0,1,1]
        d['text_bert']=t;d['audio'][0,2]=0
        x,o,l=raw_arrays(d)
        self.assertEqual(l[0],5)
        self.assertFalse(o[0,1,2]);self.assertFalse(o[:,:,5:].any())
        for xx in x:self.assertFalse(xx[:,5:].any())


if __name__=='__main__':unittest.main()
