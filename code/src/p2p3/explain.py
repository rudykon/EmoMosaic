"""Exact three-player Shapley, local occlusion and gradient*input, no retraining."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
from predict import Predictor
from data import write_json


@torch.inference_mode()
def shapley(predictor,tensors):
    x,o,l=tensors;values={}
    for subset in range(8):
        mask=torch.zeros_like(o)
        for m in range(3):
            if subset&(1<<m):mask[:,m]=True
        values[subset]=predictor.tensor_forward(tensors,mask)['y']
    phi=torch.zeros((len(l),3),device=o.device)
    for m in range(3):
        for subset in range(8):
            if subset&(1<<m):continue
            k=subset.bit_count();factor=math.factorial(k)*math.factorial(2-k)/6
            phi[:,m]+=factor*(values[subset|(1<<m)]-values[subset])
    residual=phi.sum(1)-(values[7]-values[0])
    return phi,values,residual


@torch.inference_mode()
def occlusion(predictor,tensors,width=3):
    x,o,l=tensors;original=predictor.tensor_forward(tensors)['y'];scores=torch.zeros_like(o,dtype=torch.float)
    coverage=torch.zeros_like(scores)
    for m in range(3):
        for start in range(50):
            mask=torch.ones_like(o);mask[:,m,start:start+width]=False
            p=predictor.tensor_forward(tensors,mask)['y'];delta=original-p
            scores[:,m,start:start+width]+=delta[:,None];coverage[:,m,start:start+width]+=1
    return scores/coverage.clamp_min(1)*o


def gradient_input(predictor,tensors):
    x,o,l=tensors;xx=[v.detach().clone().requires_grad_(True) for v in x]
    p=predictor.tensor_forward((xx,o,l))['y']
    grads=torch.autograd.grad(p.sum(),xx)
    return torch.stack([(g*v).sum(-1) for g,v in zip(grads,xx)],1).detach()*o


def rank_fusion(alpha,delta,grad,obs):
    from scipy.stats import rankdata
    result=np.zeros_like(alpha)
    for n in range(len(alpha)):
        for m in range(3):
            keep=obs[n,m]
            if keep.any():result[n,m,keep]=sum(rankdata(np.abs(a[n,m,keep]),method='average')/keep.sum()
                                             for a in [alpha,delta,grad])/3
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cpu');args=ap.parse_args();torch.set_num_threads(2)
    p=Predictor(device=args.device);d=p.load_file(args.input);t=p.input_tensors(d)
    result=p.predict(d);phi,values,residual=shapley(p,t);delta=occlusion(p,t);grad=gradient_input(p,t)
    fused=rank_fusion(result['alpha'],delta.cpu().numpy(),grad.cpu().numpy(),t[1].cpu().numpy())
    output={'prediction':{k:v.tolist() for k,v in result.items()},'shapley':phi.cpu().tolist(),
            'empty_value':values[0].cpu().tolist(),'shapley_efficiency_residual':residual.cpu().tolist(),
            'occlusion_delta':delta.cpu().tolist(),'gradient_input':grad.cpu().tolist(),
            'rank_fusion':fused.tolist(),'top_positions':[
                [[int(i) for i in np.argsort(-fused[n,m]) if bool(t[1][n,m,i])][:5]
                 for m in range(3)] for n in range(len(fused))],
            'position_semantics':'aligned_50 token positions; no time/frame mapping is inferred without P1 alignment.'}
    write_json(args.output,output);print(args.output)


if __name__=='__main__':main()
