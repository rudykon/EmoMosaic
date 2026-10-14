"""Reproducible missingness; masks describe permitted observations, never padding."""
import numpy as np

TYPES = ('T','A','V','TA','TV','AV','TAV')


def interval_mask(length, count, segments, rng, location=None):
    z = np.zeros(50, bool)
    count = min(length, max(0, int(count)))
    if not count: return z
    if count == length: z[:length] = True; return z
    segments = min(segments, count, length-count+1)
    if segments == 1:
        start = int(rng.integers(length-count+1)) if location is None else int(round(location*(length-count)))
        z[start:start+count] = True
        return z
    cuts = np.sort(rng.choice(np.arange(1,count),segments-1,replace=False))
    sizes = np.diff(np.r_[0,cuts,count])
    # Uniform weak composition of remaining zeros with mandatory internal gaps.
    spare = length-count-(segments-1)
    bars = np.sort(rng.choice(np.arange(spare+segments),segments,replace=False))
    gaps = np.diff(np.r_[-1,bars,spare+segments])-1
    gaps[1:-1] += 1
    start = gaps[0]
    for k,s in enumerate(sizes):
        z[start:start+s] = True; start += s+gaps[k+1]
    return z


def fixed_mask(lengths, kind='TAV', rate=.4, seed=201, segments=1,
               location=None, count=None, colocated=False, point=False):
    rng=np.random.default_rng(seed);keep=np.ones((len(lengths),3,50),bool)
    for n,L in enumerate(lengths):
        k=min(int(L),int(count) if count is not None else int(np.ceil(rate*L)))
        shared=None
        for m,name in enumerate('TAV'):
            if name not in kind: continue
            if point:
                z=np.zeros(50,bool);z[rng.choice(L,k,replace=False)]=True
            elif colocated and shared is not None:z=shared
            else:z=interval_mask(int(L),k,segments,rng,location);shared=z
            keep[n,m,z]=False
        keep[n,:,L:]=False
    return keep


def training_mask(lengths, rng, epoch, epochs, point=False, clean=False, cfg=None):
    cfg=cfg or {}
    keep=np.ones((len(lengths),3,50),bool)
    initial=cfg.get('curriculum_initial_cap',.3)
    cap=min(1.,initial+(1-initial)*epoch/max(1,epochs*cfg.get('curriculum_fraction',.3)))
    for n,L in enumerate(lengths):
        keep[n,:,L:]=False
        if clean or rng.random()<cfg.get('clean_probability',.2):continue
        # Explicit 1% all-missing training supports a meaningful Shapley empty coalition.
        if rng.random()<cfg.get('all_missing_probability_given_corrupt',.01):keep[n,:,:L]=False;continue
        num=rng.choice(cfg.get('missing_modality_counts',[1,2,3]),p=cfg.get('missing_modality_probabilities',[.5,.35,.15])); mods=rng.choice(3,num,replace=False)
        colocated=rng.random()<cfg.get('colocated_probability',.3);shared=None
        for m in mods:
            u=rng.random()
            probabilities=cfg.get('missing_rate_probabilities',[.7,.25,.05])
            rate=rng.uniform(*cfg.get('missing_rate_low_range',[.1,.6])) if u<probabilities[0] else rng.uniform(*cfg.get('missing_rate_high_range',[.6,.95])) if u<sum(probabilities[:2]) else 1.
            k=int(np.ceil(min(rate,cap)*L));segments=1 if rng.random()<1-cfg.get('two_segment_probability',.2) else 2
            if point:
                z=np.zeros(50,bool);z[rng.choice(L,k,replace=False)]=True
            elif colocated and shared is not None:z=shared
            else:z=interval_mask(int(L),k,segments,rng);shared=z
            keep[n,m,z]=False
    return keep
