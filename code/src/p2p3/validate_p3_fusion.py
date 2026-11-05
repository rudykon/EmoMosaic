"""Literal word deletion contrast, with all six text encoders recomputed."""
import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel
from data import ROOT, write_json
from p3_common import OUT3, dataset, token_mapping, ordering, corr
from p3_fusion import FusionExplainer


def real_deletion(device='cpu'):
    output=OUT3/'validation/real_word_deletion.json'
    if output.exists():return
    p=FusionExplainer(device);d=dataset('valid');p.prepare(d,'valid')
    with np.load(OUT3/'explanations/valid_fusion.npz') as z:e=dict(z)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
    cases=[];indices=[];masks=[];tokens=[];audio=[];vision=[];observers=[];raws=[]
    for i,raw in enumerate(d['raw_text']):
        mapping=token_mapping(str(raw),d['token_ids'][i],tok)
        c=int(e['probs'][i].argmax())
        order=ordering(e['occlusion_probs'][i,0,:,c],e['eligible'][i,0])
        selected=next((mapping[int(pos)]['word_index'] for pos in order if mapping[int(pos)]['word_index'] is not None),None)
        if selected is None:continue
        words=list(re.finditer(r'\S+',str(raw)))
        remain=[j for j in range(len(words)) if j!=selected]
        changed=' '.join(words[j].group() for j in remain)
        enc=tok(changed,truncation=True,max_length=50,padding='max_length')
        mp=token_mapping(changed,np.asarray(enc['input_ids']),tok)
        av=[np.zeros((50,d[m].shape[-1]),np.float32) for m in ['audio','vision']]
        oo=np.zeros((3,50),bool);oo[0]=np.asarray(enc['attention_mask'],bool);tail=0
        for item in mp:
            if item['word_index'] is None:continue
            old_wi=remain[item['word_index']]
            old=next((q['position'] for q in mapping if q['word_index']==old_wi),None)
            if old is None:tail+=1;continue
            for j,m in enumerate(['audio','vision']):
                av[j][item['position']]=d[m][i,old];oo[j+1,item['position']]=d['obs'][i,j+1,old]
        positions=[q['position'] for q in mapping if q['word_index']==selected]
        mask=np.ones((3,50),bool);mask[:,positions]=False
        indices.append(i);masks.append(mask);tokens.append([enc['input_ids'],enc['attention_mask'],enc['token_type_ids']])
        audio.append(av[0]);vision.append(av[1]);observers.append(oo);raws.append(changed)
        cases.append(dict(id=str(d['ids'][i]),word=words[selected].group(),word_index=selected,
                          positions=json.dumps(positions),tail_unobserved_positions=tail))
    masked=p.forward(indices,masks)
    tokens=np.asarray(tokens);oo=np.asarray(observers)
    cache=OUT3/'fusion_cache/real_deletion_rm_text.npy'
    if cache.exists():text=np.load(cache)
    else:
        model=BertModel.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True,use_safetensors=True).to(device).eval()
        pieces=[]
        with torch.inference_mode():
            for start in range(0,len(tokens),24):
                b=torch.as_tensor(tokens[start:start+24],device=device)
                h=model(input_ids=b[:,0],attention_mask=b[:,1],token_type_ids=b[:,2]).last_hidden_state.cpu().numpy()
                pieces.append(h)
        st=p.public.original.stats['text']
        text=np.clip((np.concatenate(pieces)-np.asarray(st['mean'],np.float32))/np.asarray(st['std'],np.float32),-5,5)
        text*=oo[:,0,:,None];np.save(cache,text)
        del model
    changed=dict(text=text,audio=np.asarray(audio),vision=np.asarray(vision),obs=oo,
                 token_ids=tokens[:,0],lengths=tokens[:,1].sum(1),raw_text=np.asarray(raws),ids=d['ids'][indices])
    p.prepare(changed,'real_deletion');deleted=p.forward(np.arange(len(indices)))
    c=e['probs'][indices].argmax(1);base=e['probs'][indices,c]
    for j,row in enumerate(cases):
        row.update(original_intensity=float(e['y'][indices[j]]),
            mask_intensity_delta=float(e['y'][indices[j]]-masked['y'][j]),
            delete_intensity_delta=float(e['y'][indices[j]]-deleted['y'][j]),
            mask_probability_delta=float(base[j]-masked['probs'][j,c[j]]),
            delete_probability_delta=float(base[j]-deleted['probs'][j,c[j]]))
    frame=pd.DataFrame(cases);frame.to_csv(OUT3/'validation/real_word_deletion.csv',index=False)
    stats=dict(n_samples=len(d['ids']),n_pairs=len(frame),words_per_sample=1,
               tail_affected_pairs=int((frame.tail_unobserved_positions>0).sum()),
               method='Highest signed classification-occlusion text word per sample; delete the full raw word, re-encode core BERT and all five LoRA encoders; shift official AV rows and mark newly exposed tail rows unobserved.',
               limitation='Contextual leakage, position shift, and reliability denominator changes are combined; descriptive sensitivity contrast only.')
    for target in ['intensity','probability']:
        a=frame['mask_'+target+'_delta'];b=frame['delete_'+target+'_delta']
        stats[target]=dict(spearman=corr(a,b),absolute_spearman=corr(abs(a),abs(b)),mean_absolute_difference=float(abs(a-b).mean()))
    write_json(output,stats);print(json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(args.threads);real_deletion(args.device)
