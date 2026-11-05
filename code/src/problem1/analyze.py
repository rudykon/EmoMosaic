"""Descriptive QC and figures computed solely from the provided 100 samples."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from core import (ROOT, DATA, probe, waveform, sampled_frames, save_json,
                  interval_frame_indices, word_sample_bounds)

OUT=ROOT/'outputs/problem1'
ANALYSIS=OUT/'analysis'
sys.path.insert(0,str(ROOT/'src/common'))
from plot_style import COLORS, INTERVAL_COLORS, DIVERGING, apply_style, boxplot, save_figure
apply_style()

def retention(x, groups):
    if not len(x):return None
    variance=x.std(axis=0);keep=variance>1e-8
    z=(x[:,keep]-x[:,keep].mean(axis=0))/variance[keep]
    total=np.sum(z*z)
    if total<1e-8:return None
    residual=0
    for g in np.unique(groups):
        p=z[groups==g];residual+=np.sum((p-p.mean(axis=0))**2)
    return float(1-residual/total)

def analyze():
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(OUT/'manifest.csv',keep_default_na=False);rows=[]
    for _,r in m.iterrows():
        sid=r.sample_id;a=np.load(OUT/'L0_frame'/f'{sid}.audio.npz');v=np.load(OUT/'L0_frame'/f'{sid}.vision.npz')
        words=json.loads((OUT/'word_times'/f'{sid}.json').read_text())
        groups=np.full(len(a['audio']),-1,np.int32)
        for w in words:
            if w['timing_usable']:
                groups[interval_frame_indices(a['audio_times'],*word_sample_bounds(w))]=w['i']
        active=groups>=0;x=a['audio'][active].astype(np.float64);at=a['audio_times'][active]
        item=dict(sample_id=sid,coverage_word_audio=float(active.mean()),
            R_word=retention(x,groups[active]),source_video_max_error_ms=float(np.max(np.abs(v['source_video_pts']-v['vision_times']))*1000))
        for delta in [.25,.5,1.]:item['R_'+str(delta)]=retention(x,np.floor(at/delta).astype(int))
        item['n_groups_word']=len(np.unique(groups[active]))
        for delta in [.25,.5,1.]:item['n_groups_'+str(delta)]=len(np.unique(np.floor(at/delta)))
        vt=v['vision_times'];energy=np.interp(vt,a['audio_times'],a['audio'][:,4]);mouth=v['vision'][:,32]
        corr=[]
        for lag in range(-12,13):
            ai=np.arange(len(vt));vi=ai+lag;ok=(vi>=0)&(vi<len(vt));ai=ai[ok];vi=vi[ok]
            ok=v['face_mask'][vi];aa=energy[ai[ok]];bb=mouth[vi[ok]]
            value=np.corrcoef(aa,bb)[0,1] if len(aa)>=25 and aa.std()>1e-6 and bb.std()>1e-6 else np.nan
            corr.append(value)
        if np.isfinite(corr).any():
            best=int(np.nanargmax(corr));item.update(lag_ms=(best-12)*40,lag_correlation=float(corr[best]))
        else:item.update(lag_ms=None,lag_correlation=None)
        item['word_voiced_fraction']=float(np.mean([a['audio'][w['a_frames'],3].mean()>.5 for w in words]))
        item['nonword_energy_ratio']=float(a['audio'][active,4].mean()/(a['audio'][~active,4].mean()+1e-12)) if active.any() and (~active).any() else None
        rows.append(item)
    q=pd.DataFrame(rows);q.to_csv(ANALYSIS/'qc_metrics.csv',index=False)
    official=json.loads((ANALYSIS/'official_audit.json').read_text())
    hits=[h for h in official['matches'] if h['match']]
    cross=[]
    for h in hits:
        r=m[m.sample_id==h['id']].iloc[0]
        cross.append(dict(sample_id=h['id'],split=h['match'][0],label_difference=float(r.label-h['official_label']),
            p1_words=int(r.n_words),p1_bert_tokens=int(r.n_bert_tokens),official_tokens=h['official_token_length'],
            text_exact_match=r.text==h['official_text']))
    pd.DataFrame(cross).to_csv(ANALYSIS/'official_crosscheck.csv',index=False)
    summary=dict(n_samples=len(m),n_source_videos=m.video_id.nunique(),duration_total=float(m.duration.sum()),
        duration_min=float(m.duration.min()),duration_max=float(m.duration.max()),duration_mean=float(m.duration.mean()),
        words_total=int(m.n_words.sum()),words_min=int(m.n_words.min()),words_max=int(m.n_words.max()),
        audio_frames=int(m.n_audio.sum()),video_frames=int(m.n_vision.sum()),
        n_truncated=int(m.truncated.sum()),n_av_duration_difference=int(m['flags'].str.contains('av_duration_difference').sum()),
        n_low_face=int(m['flags'].str.contains('low_face').sum()),n_text_mismatch=int(m['flags'].str.contains('text_mismatch').sum()),
        n_low_alignment=int(m['flags'].str.contains('low_alignment_confidence').sum()),
        face_rate_frame_weighted=float(np.average(m.face_rate,weights=m.n_vision)),
        word_vision_rate=float(np.average(m.word_vision_rate,weights=m.n_words)),
        align_conf_mean=float(m.align_conf.mean()),wer_median=float(m.wer_ctc.median()),
        timing_usable_words=int(m.timing_usable_words.sum()),timing_usable_samples=int((m.timing_usable_words>0).sum()),
        audio_fallbacks=int(m.audio_fallbacks.sum()),video_fallbacks=int(m.video_fallbacks.sum()),
        nonfinite_replaced=int(m.audio_nonfinite_replaced.sum()),official_matches=len(hits),
        labels=m.annotation.value_counts().to_dict(),
        retention_means={k:float(q[k].mean()) for k in ['R_word','R_0.25','R_0.5','R_1.0']},
        retention_group_means={k:float(q[k].mean()) for k in q if k.startswith('n_groups_')},
        word_audio_coverage_mean=float(q.coverage_word_audio.mean()),
        source_video_max_error_ms=float(q.source_video_max_error_ms.max()),
        text_quantization_max_abs=float(m.text_quantization_max_abs.max()),
        lag_median_abs_ms=float(q.lag_ms.abs().median()),
        lag_within_40ms=int((q.lag_ms.abs()<=40).sum()),lag_eligible=int(q.lag_ms.notna().sum()),
        cpu_extraction_seconds=float(pd.to_numeric(m.audio_text_seconds).sum()+pd.to_numeric(m.vision_seconds).sum()))
    save_json(ANALYSIS/'summary.json',summary)
    qc_figure(m,q)
    # Select the clearly stated rule before inspecting individual visualizations.
    candidates=m[(m.annotation=='Positive')&(m.n_words.between(8,24))&(m.face_rate>.9)].copy()
    typical=candidates.sort_values(['wer_ctc','align_conf'],ascending=[True,False]).iloc[0]
    save_json(ANALYSIS/'typical_selection.json',dict(sample_id=typical.sample_id,
        rule='Positive, 8–24 words, face_rate > 0.9; lowest greedy-ASR WER, then highest CTC confidence.'))
    typical_figure(typical)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def qc_figure(m,q):
    fig,ax=plt.subplots(2,3,figsize=(10,5.5))
    for a,col,label,c in [(ax[0,0],'duration','Common duration (s)',0),(ax[0,1],'n_words','Words per sample',1),
                           (ax[0,2],'face_rate','Detected-face fraction',2),(ax[1,0],'align_conf','Mean CTC score',0)]:
        a.hist(m[col],bins=15,color=COLORS[c],edgecolor='black');a.set_xlabel(label);a.set_ylabel('Samples')
    methods=['R_word','R_0.25','R_0.5','R_1.0']
    boxplot(ax[1,1],[q[k].dropna() for k in methods],['Word','.25 s','.5 s','1 s'])
    ax[1,1].set_ylabel('Retention on identical speech frames')
    ax[1,2].hist(q.lag_ms.dropna(),bins=np.arange(-500,521,40),color=COLORS[4],edgecolor='black')
    ax[1,2].set_xlabel('Exploratory energy / mouth lag (ms)');ax[1,2].set_ylabel('Samples')
    fig.tight_layout(pad=1.4);save_figure(fig,OUT/'figures','qc_summary')

def typical_figure(r):
    sid=r.sample_id;a=np.load(OUT/'L0_frame'/f'{sid}.audio.npz');v=np.load(OUT/'L0_frame'/f'{sid}.vision.npz')
    l=np.load(OUT/'L1_word'/f'{sid}.npz');words=json.loads((OUT/'word_times'/f'{sid}.json').read_text())
    # Historical manifests contain paths from before the data directory was
    # renamed. Resolve by immutable sample IDs, within this project's data.
    media=DATA/str(r.video_id)/(str(int(r.clip_id))+'.mp4')
    y=waveform(media,probe(media));t=np.arange(len(y))/16000
    fig,axs=plt.subplots(5,1,figsize=(11,9),sharex=True,gridspec_kw={'height_ratios':[1.5,1,1,1.1,1.1]})
    axs[0].plot(t[::20],y[::20],color=COLORS[0],lw=.5)
    top=max(float(np.max(np.abs(y))),.01)
    for i,w in enumerate(words):
        axs[0].axvspan(w['s'],w['e'],facecolor=INTERVAL_COLORS[i%2],
                       edgecolor='#808080',linewidth=.4,zorder=.5)
        axs[0].text((w['s']+w['e'])/2,top*(1.15+.5*(i%2)),w['word'],rotation=65,fontsize=7,ha='center',va='bottom')
    axs[0].set_ylim(-top,top*3.8);axs[0].set_ylabel('Waveform / words');axs[0].set_title(sid+'  |  '+r.annotation,fontsize=10,parse_math=False)
    axs[1].plot(a['audio_times'],a['audio'][:,0],color=COLORS[0],label='F0 (Hz)');axs[1].set_ylabel('F0 (Hz)')
    twin=axs[1].twinx();twin.plot(a['audio_times'],a['audio'][:,4],color=COLORS[1],label='RMS');twin.set_ylabel('RMS')
    for idx,label,c in [(13,'Jaw open',0),(14,'Smile left',1),(0,'Brow down left',2)]:
        values=v['vision'][:,idx].copy();values[~v['face_mask']]=np.nan
        axs[2].plot(v['vision_times'],values,label=label,color=COLORS[c],lw=1)
    axs[2].legend(ncol=3,loc='upper right',fontsize=7);axs[2].set_ylabel('Blendshape score')
    vt=v['vision_times'];pc=PCA(n_components=min(8,len(words)),svd_solver='full').fit_transform(l['text'].astype(np.float32))
    compact=np.column_stack([l['audio'][:,[0,4,8,14]],l['vision'][:,[13,14,0,32]]])
    for ax,values,label in [(axs[3],pc,'BERT PCA components'),(axs[4],compact,'Audio / visual channels')]:
        values=(values-values.mean(axis=0))/(values.std(axis=0)+1e-6)
        grid=np.full((values.shape[1],len(vt)),np.nan)
        for i,w in enumerate(words):
            grid[:,interval_frame_indices(vt,*word_sample_bounds(w))]=values[i,:,None]
        ax.imshow(grid,aspect='auto',origin='lower',extent=[0,len(vt)/25,0,len(grid)],cmap=DIVERGING,vmin=-2.5,vmax=2.5)
        ax.set_ylabel(label);ax.grid(False)
    axs[-1].set_xlabel('Time from common stream start (s)');axs[-1].set_xlim(0,r.duration)
    fig.tight_layout(pad=1.4);save_figure(fig,OUT/'figures','typical_alignment')
    table=[]
    for w in words:
        i=w['i'];table.append(dict(word=w['word'],start_s=w['s'],end_s=w['e'],confidence=w['confidence'],
            audio_frames=f"{min(w['a_frames'])}–{max(w['a_frames'])}",
            video_grid_frames=f"{min(w['v_frames'])}–{max(w['v_frames'])}",
            source_video_frames=','.join(map(str,w['source_video_frames'])),
            bert_subtokens=','.join(map(str,w['bert_subtokens'])),
            F0_Hz=float(l['audio'][i,0]),RMS=float(l['audio'][i,4]),
            jaw_open=float(l['vision'][i,13]),smile_left=float(l['vision'][i,14])))
    if '--figures-only' not in sys.argv:
        pd.DataFrame(table).to_csv(ANALYSIS/'typical_correspondence.csv',index=False)
    targets=np.linspace(.1,max(.1,r.duration-.1),6);frames=list(sampled_frames(media,probe(media)))
    fig,axs=plt.subplots(1,6,figsize=(11,2.1))
    for ax,target in zip(axs,targets):
        selected=min(frames,key=lambda f:abs(f[0]-target));tt,pts,idx,rgb=selected
        ax.imshow(rgb);ax.set_title(f'PTS {pts:.3f} s\nsource frame {idx}',fontsize=7);ax.axis('off')
    fig.tight_layout();save_figure(fig,OUT/'figures','typical_frames')

if __name__=='__main__':
    if '--figures-only' in sys.argv:
        m=pd.read_csv(OUT/'manifest.csv',keep_default_na=False)
        qc_figure(m,pd.read_csv(ANALYSIS/'qc_metrics.csv'))
        sid=json.loads((ANALYSIS/'typical_selection.json').read_text())['sample_id']
        typical_figure(m[m.sample_id==sid].iloc[0])
    else:
        analyze()
