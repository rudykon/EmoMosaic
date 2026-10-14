"""One raw-feature interface for P2/P3, optional arbitrary observation masks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import ROOT,OUT,MODALITIES,load_pickle,unwrap,raw_arrays,normalize,zero_runs
from model import RMNet
from metrics import classes


class Predictor:
    def __init__(self,model_dir=None,device='cpu',ensemble=True,cache_dir=None,load_models=True):
        self.folder=Path(model_dir) if model_dir else ROOT/'models/rmnet'
        self.cache_folder=Path(cache_dir) if cache_dir else OUT/'cache'
        self.device=torch.device(device);self.stats=json.loads((self.folder/'norm_stats.json').read_text())
        self.rules=json.loads((self.folder/'decision_rules.json').read_text());self.models=[];self.bert=None
        names=([f'weights_seed{s}_fp16.pt' for s in self.rules['ensemble_seeds']] if ensemble else ['weights_fp16.pt']) if load_models else []
        self.thresholds=self.rules['thresholds'] if ensemble else self.rules['best_single_thresholds']
        for name in names:
            ck=torch.load(self.folder/name,map_location='cpu',weights_only=False)
            net=RMNet(ck['config'],ck['variant']);net.load_state_dict(ck['state_dict']);net.to(self.device).eval()
            self.models.append(net)
        self.config=ck['config'] if self.models else {};self.cached_text={}
        for filename in ['attachment3_text_cache.npz','attachment4_text_cache.npz']:
            path=self.cache_folder/filename
            if path.exists():
                d=np.load(path,allow_pickle=False)
                for h,x in zip(d['sha256'],d['text']):self.cached_text[str(h)]=x

    @torch.inference_mode()
    def encode_text(self,d):
        from transformers import BertModel
        if self.bert is None:
            self.bert=BertModel.from_pretrained(ROOT/'models/pretrained/bert',
                         local_files_only=True,use_safetensors=True).to(self.device).eval()
        t=np.asarray(d['text_bert']);t=t[None] if t.ndim==2 else t
        b=torch.as_tensor(t,dtype=torch.long,device=self.device)
        observed=(b[:,1]!=0)&(b[:,0]!=0);safe=observed.clone();safe[~safe.any(1),0]=True
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):
            y=self.bert(input_ids=b[:,0],attention_mask=safe,token_type_ids=b[:,2]).last_hidden_state
        return (y.float()*observed[...,None]).cpu().numpy()

    def load_file(self,path):
        path=Path(path);d=dict(unwrap(load_pickle(path)));digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in self.cached_text:
            # Cached values are normalized; invert to make the public raw-input path uniform.
            t=np.asarray(d['text_bert']);t=t[None] if t.ndim==2 else t
            observed=(t[:,1]!=0)&(t[:,0]!=0)
            d['text']=((self.cached_text[digest].astype(np.float32)[None]*np.asarray(self.stats['text']['std'],np.float32)
                       +np.asarray(self.stats['text']['mean'],np.float32))*observed[...,None])
        else:d['text']=self.encode_text(d)
        return d

    def input_tensors(self,X):
        d=dict(unwrap(X))
        if 'text' not in d:d['text']=self.encode_text(d)
        arrays,obs,lengths=raw_arrays(d)
        xx=normalize(arrays,obs,self.stats)
        return ([torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in xx],
                torch.as_tensor(obs,device=self.device),torch.as_tensor(lengths,device=self.device))

    def tensor_forward(self,tensors,mask=None):
        if not self.models:raise RuntimeError('This Predictor was initialized for feature loading only')
        x,o,l=tensors
        if mask is not None:
            mask=torch.as_tensor(mask,dtype=torch.bool,device=self.device)
            if mask.ndim==2:mask=mask[None]
            if tuple(mask.shape)!=tuple(o.shape):raise ValueError(f'mask must be {tuple(o.shape)}, got {tuple(mask.shape)}')
        parts=[model(x,o,l,mask) for model in self.models]
        result={k:torch.stack([p[k].float() for p in parts]).mean(0) for k in
                ['y','w','y_m','alpha','recon_err','y_int','rho','z','contributions']}
        result['ensemble_std']=torch.stack([p['y'].float() for p in parts]).std(0,unbiased=False)
        # Ensemble contributions MUST be mean(w_m*y_m), not mean(w_m)*mean(y_m).
        # Return effective branch estimates so the documented identity also holds for the ensemble.
        result['y_m_individual_mean']=result['y_m']
        result['y_m']=torch.where(result['w']>0,result['contributions']/result['w'].clamp_min(1e-9),torch.zeros_like(result['w']))
        result['auxiliary_probs']=torch.stack([p['logits'].float().softmax(-1) for p in parts]).mean(0)
        temperature=self.rules.get('ordinal_temperature',.25)
        pneg=torch.sigmoid((-self.thresholds[0]-result['y'])/temperature)
        ppos=torch.sigmoid((result['y']-self.thresholds[1])/temperature)
        result['probs']=torch.stack([pneg,(1-pneg-ppos).clamp_min(0),ppos],-1)
        return result

    @torch.inference_mode()
    def predict(self,X,mask=None):
        tensors=self.input_tensors(X)
        result=self.tensor_forward(tensors,mask)
        result={k:v.cpu().numpy() for k,v in result.items()}
        result['polarity']=np.asarray(['Negative','Neutral','Positive'])[classes(result['y'],self.thresholds)]
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--single',action='store_true')
    ap.add_argument('--improvement-config',help='Frozen probability-fusion decision configuration')
    ap.add_argument('--legacy',action='store_true',help='Use the historical intensity-threshold interface explicitly')
    args=ap.parse_args()
    if args.legacy and args.improvement_config:ap.error('--legacy conflicts with --improvement-config')
    deployed=ROOT/'models/rmnet_bert_lora/decision_rules.json'
    if not args.improvement_config and not args.legacy and not args.single and deployed.exists():
        args.improvement_config=str(deployed)
    if args.improvement_config:
        if args.single:ap.error('--single cannot be combined with --improvement-config')
        from improved_predict import export_csv
        torch.set_num_threads(2)
        export_csv(args.improvement_config,args.input_dir,args.output,args.device)
        return
    torch.set_num_threads(2);predictor=Predictor(device=args.device,ensemble=not args.single);rows=[]
    for path in sorted(Path(args.input_dir).glob('*.pkl')):
        d=predictor.load_file(path);arrays,obs,lengths=raw_arrays(d);p=predictor.predict(d)
        if len(lengths)!=1:raise ValueError('CSV file-per-sample entry must contain one sample')
        row={'file':path.name,'sample_id':str(d.get('id',path.stem)),'valid_len':int(lengths[0]),
             'pred_intensity':float(p['y'][0]),'pred_polarity':p['polarity'][0],
             'ensemble_std':float(p['ensemble_std'][0]),'y_int':float(p['y_int'][0])}
        for j,m in enumerate(MODALITIES):
            row[f'miss_{m}_intervals']=json.dumps(zero_runs(~obs[0,j,:lengths[0]]),separators=(',',':'))
            row[f'miss_ratio_{"tav"[j]}']=float(1-obs[0,j].sum()/lengths[0])
            row[f'w_{m}']=float(p['w'][0,j]);row[f'contribution_{m}']=float(p['contributions'][0,j])
        for j,label in enumerate(['neg','neu','pos']):
            row[f'p_{label}']=float(p['probs'][0,j]);row[f'aux_p_{label}']=float(p['auxiliary_probs'][0,j])
        rows.append(row)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError(f'No .pkl files found in {args.input_dir}')
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} predictions: {output}')


if __name__=='__main__':main()
