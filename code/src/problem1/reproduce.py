"""Compare two independent raw-media extractions, including actual file MD5s."""
from pathlib import Path
import json
import numpy as np
from core import ROOT, digest, save_json

def main():
    a=ROOT/'outputs/problem1';b=a/'validation/repro';results=[]
    if not all((b/sub).is_dir() for sub in ('L0_frame','L1_word','word_times')):
        raise SystemExit('Independent extraction features are absent or were cleaned. '
                         'Run bash code/src/problem1/reproduce_all.sh from the project root to extract both runs again; '
                         'the saved reproducibility.json remains the historical comparison.')
    for sub,suffix in [('L0_frame','*.npz'),('L1_word','*.npz'),('word_times','*.json')]:
        for p in sorted((a/sub).glob(suffix)):
            q=b/sub/p.name
            assert q.exists(),q
            delta=0.
            if p.suffix=='.npz':
                x=np.load(p);y=np.load(q);assert x.files==y.files
                for k in x.files:
                    assert x[k].shape==y[k].shape
                    delta=max(delta,float(np.max(np.abs(x[k].astype(np.float64)-y[k].astype(np.float64)),initial=0)))
            else:assert json.loads(p.read_text())==json.loads(q.read_text())
            results.append(dict(file=str(p.relative_to(a)),max_abs_difference=delta,
                md5_first=digest(p,'md5'),md5_second=digest(q,'md5')))
    assert len(results)==500,len(results)
    summary=dict(files_compared=len(results),samples=100,
        max_abs_difference=max(x['max_abs_difference'] for x in results),
        byte_identical=sum(x['md5_first']==x['md5_second'] for x in results),files=results)
    save_json(a/'reproducibility.json',summary)
    assert summary['max_abs_difference']==0,summary['max_abs_difference']
    print({k:v for k,v in summary.items() if k!='files'})

if __name__=='__main__':main()
