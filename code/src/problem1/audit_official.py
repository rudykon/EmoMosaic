"""Inspect attachment2 without using its features as our extraction outputs."""
import json
import pickle
import numpy as np
from core import ROOT, SOURCE_DATA, records, save_json

def main():
    p=SOURCE_DATA/'attachment2_features/aligned_50.pkl';data=pickle.load(p.open('rb'));meta={};ids={};examples={}
    for split,s in data.items():
        meta[split]={k:{'shape':list(v.shape),'dtype':str(v.dtype)} if hasattr(v,'shape') else {'len':len(v),'type':type(v).__name__} for k,v in s.items()}
        ids.update({str(sid):(split,i) for i,sid in enumerate(s['id'])})
        examples[split]={'id':str(s['id'][0]),'text_bert':s['text_bert'][0].tolist(),
            'nonzero_positions':{k:np.flatnonzero(np.any(s[k][0]!=0,axis=-1)).tolist() for k in ['text','audio','vision']}}
    matches=[]
    for row in records():
        sid=row['sample_id'];hit=ids.get(sid);item={'id':sid,'match':hit}
        if hit:
            s=data[hit[0]];i=hit[1]
            item.update(official_text=s['raw_text'][i],official_token_length=int(s['text_bert'][i,1].sum()),
                official_nonzero={k:int(np.any(s[k][i]!=0,axis=-1).sum()) for k in ['text','audio','vision']},
                official_label=np.asarray(s['regression_labels'][i]).tolist())
        matches.append(item)
    save_json(ROOT/'outputs/problem1/analysis/official_audit.json',dict(schema=meta,matches=matches,examples=examples))
    print('matched',sum(m['match'] is not None for m in matches))

if __name__=='__main__':main()
