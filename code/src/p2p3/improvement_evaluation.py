"""Cached-feature inference for frozen candidate configurations and mask grids."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax

from data import load_split, ROOT, MODALITIES
from project_paths import resolve_asset
from model import RMNet
from train import device_data, infer
from bert_layer_data import load_layer_split
from probe_text_experts import pooled
from boosted_classifier import sequence_features
from improvement_common import EXP, digest, masked, apply_bias, fusion, text_observed_fraction


class CachedEvaluator:
    def __init__(self,threads=2):
        torch.set_num_threads(threads)
        self.data={};self.layers={};self.heads={};self.models={};self.hashes={};self.adapter_hidden={}

    def file_digest(self,path):
        path=Path(path);stat=path.stat()
        key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        if key not in self.hashes:self.hashes[key]=digest(path)
        return self.hashes[key]

    def dataset(self,split):
        if split not in self.data:self.data[split]=load_split(split)
        return self.data[split]

    def layer_dataset(self,split):
        if split not in self.layers:self.layers[split]=load_layer_split(split)
        return self.layers[split]

    @staticmethod
    def resolve(config,path):
        # Frozen experiment records may retain the old absolute base directory.
        # Resolve their named assets inside the relocated standalone project.
        return resolve_asset(path)

    def core(self,config,split,mask=None):
        d=self.dataset(split)
        effective=d['obs'] if mask is None else d['obs']&mask
        files=[self.resolve(config,p) for p in config['core_checkpoints']]
        signature=[self.file_digest(p) for p in files]
        signature.append(hashlib.sha256(effective.tobytes()).hexdigest())
        signature.append(self.file_digest(ROOT/f'outputs/problem2/cache/{split}.npz'))
        if 'layer_norm_stats' in config:
            signature.append(self.file_digest(self.resolve(config,config['layer_norm_stats'])))
            signature.append(json.loads((EXP/'features/bert'/split/'complete.json').read_text())['hidden_sha256'])
        key=hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        path=EXP/'core_prediction_cache'/f'{split}_{key}.npz'
        if path.exists():
            with np.load(path) as z:
                if not np.array_equal(z['ids'],d['ids']):raise ValueError('Cached IDs differ')
                return {'probabilities':z['probabilities'],'intensity':z['intensity']}
        results=[]
        for checkpoint_path in files:
            key=(str(checkpoint_path),self.file_digest(checkpoint_path))
            if key not in self.models:
                ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                model=RMNet(ck['config'],ck['variant']);model.load_state_dict(ck['state_dict']);model.eval()
                self.models[key]=model
            model=self.models[key]
            inputs=self.layer_dataset(split) if model.cfg.get('text_layers') else d
            p=infer(model,device_data(inputs,'cpu'),torch.as_tensor(effective))
            results.append((softmax(p['logits'],axis=-1),p['y']))
        p=np.mean([v[0] for v in results],axis=0)
        y=np.mean([v[1] for v in results],axis=0)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,ids=d['ids'],probabilities=p,intensity=y)
        return {'probabilities':p,'intensity':y}

    @torch.inference_mode()
    def adapter_prediction(self,model,adapter_path,model_dir,split,mask=None):
        """Cache exact float32 states; all masking still occurs after encoding.

        This is equivalent to predict_adapter under the declared protocol. It
        cannot be used for strict deletion, which changes the encoder input.
        """
        from improvement_common import save_json
        d=self.dataset(split)
        identity={'adapter_sha256':self.file_digest(adapter_path),
                  'base_sha256':self.file_digest(Path(model_dir)/'model.safetensors'),
                  'input_sha256':self.file_digest(ROOT/f'outputs/problem2/cache/{split}.npz'),
                  'dtype':'float32','protocol':'lora_encode_original_then_mask_v1'}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        folder=EXP/'lora_hidden_cache'/key
        if key not in self.adapter_hidden:
            folder.mkdir(parents=True,exist_ok=True)
            path=folder/'hidden.npy';manifest=folder/'complete.json'
            if manifest.exists():
                saved=json.loads(manifest.read_text())
                if saved['identity']!=identity or self.file_digest(path)!=saved['hidden_sha256']:
                    raise RuntimeError('LoRA hidden cache integrity failure')
            else:
                temporary=folder/'hidden.partial.npy'
                hidden=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,
                                                shape=(*d['token_ids'].shape,model.encoder.config.hidden_size))
                for start in range(0,len(d['lengths']),32):
                    ids=torch.as_tensor(d['token_ids'][start:start+32],dtype=torch.long)
                    obs=torch.as_tensor(d['obs'][start:start+32,0],dtype=torch.bool)
                    hidden[start:start+32]=model.encode(ids,obs).numpy()
                hidden.flush();del hidden
                temporary.replace(path)
                save_json(manifest,{'identity':identity,'hidden_sha256':self.file_digest(path)})
            self.adapter_hidden[key]=np.load(path,mmap_mode='r')
        hidden=self.adapter_hidden[key];pieces=[]
        for start in range(0,len(d['lengths']),32):
            keep=d['obs'][start:start+32,0]&(d['token_ids'][start:start+32]!=100)
            if mask is not None:keep=keep&mask[start:start+32,0]
            out=model.classify(torch.from_numpy(hidden[start:start+32].copy()),torch.as_tensor(keep))
            pieces.append((out['logits'].softmax(-1).numpy(),out['y'].numpy()))
        return {'probabilities':np.concatenate([p[0] for p in pieces]),'intensity':np.concatenate([p[1] for p in pieces])}

    def expert(self,config,split,mask=None):
        specification=config['expert']
        if specification['kind']=='bert_lora':
            from lora_text_expert import load_adapter
            results=[]
            for path in specification['adapters']:
                key=str(self.resolve(config,path))
                model_dir=self.resolve(config,config['bert_model_dir'])
                if key not in self.heads:self.heads[key]=load_adapter(key,model_dir)
                results.append(self.adapter_prediction(self.heads[key],key,model_dir,split,mask))
            return np.mean([r['probabilities'] for r in results],axis=0),np.mean([r['intensity'] for r in results],axis=0)
        key=str(self.resolve(config,specification['head']))
        if key not in self.heads:self.heads[key]=joblib.load(key)
        head=self.heads[key]
        d=masked(self.dataset(split),mask)
        if specification['kind']=='svd':
            x=sequence_features(d)[:,:head['width']]
            p=head['classifier'].predict_proba(head['svd'].transform(x))
            return p,None
        x=pooled(specification['encoder'],split,specification['layer'],d['obs'][:,0],specification['pooling'])
        x=head['scaler'].transform(x)
        if 'svd' in head:x=head['svd'].transform(x)
        return head['classifier'].predict_proba(x),np.clip(head['ridge'].predict(x),-3,3)

    def predict(self,config,split,mask=None):
        core=self.core(config,split,mask)
        cp,cy=core['probabilities'],core['intensity']
        setting=config['fusion']
        p,y=apply_bias(cp,setting['bias']),cy
        if config.get('expert'):
            ep,ey=self.expert(config,split,mask)
            if config['expert']['kind']=='svd':
                p=apply_bias(setting['w0']*ep+(1-setting['w0'])*cp,setting['bias'])
            else:
                f=text_observed_fraction(masked(self.dataset(split),mask))
                p,_=fusion(ep,cp,f,setting['w0'],setting['k'],setting['bias'])
                w=config.get('regression_w0',0)*f**config.get('regression_k',setting['k'])
                w=np.where(f>0,w,0.)
                y=w*ey+(1-w)*cy
        return {'probabilities':p,'intensity':y,'core_probabilities':cp,'core_intensity':cy}


def base_config():
    return {'schema_version':1,'base_dir':str(ROOT),
            'raw_input_model_dir':'models/rmnet','raw_input_cache_dir':'outputs/problem2/cache',
            'bert_model_dir':'models/pretrained/bert',
            'core_checkpoints':[f'models/rmnet/weights_seed{s}_fp16.pt' for s in [17,29,43,71,101]],
            'expert':None,'fusion':{'bias':[0.,0.,0.],'w0':0.,'k':1.},'regression_w0':0.,'regression_k':1.}
