"""Publication plots from actual saved predictions and complete evaluation CSVs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data import ROOT,OUT,MODALITIES,load_split,write_json

FIG=OUT/'figures'
sys.path.insert(0,str(ROOT/'src/common'))
from plot_style import (COLORS, POLARITY, SEQUENTIAL, DIVERGING,
                         apply_style, heatmap_text_color, save_figure)
apply_style()
TYPES=['T','A','V','TA','TV','AV','TAV']


def save(fig,name):
    save_figure(fig,FIG,name)


def load_results():
    paths=sorted((OUT/'evaluation').glob('*/*.csv'))
    frames=[pd.read_csv(path) for path in paths]
    df=pd.concat(frames,ignore_index=True)
    df['stratum']=df.get('stratum',pd.Series(index=df.index,dtype=object)).fillna('all')
    df.to_csv(OUT/'missing_grid_raw.csv',index=False)
    return df


def summary_tables(df):
    allrows=df[df.stratum=='all'];rr=[]
    for (split,variant,seed),part in allrows.groupby(['split','variant','model_seed']):
        clean=part[part.family=='clean'].iloc[0]
        for kind in TYPES:
            q=part[(part.family=='rate')&(part.kind==kind)].groupby('rate')[['mae','macro_f1','accuracy','corr']].mean()
            q.loc[0]=[clean[k] for k in q.columns];q=q.sort_index()
            for metric in ['mae','macro_f1','accuracy','corr']:
                baseline=clean[metric]
                if metric=='corr' and baseline<=0:continue
                ratio=baseline/q[metric].clip(lower=1e-8) if metric=='mae' else q[metric]/max(baseline,1e-8)
                slope=np.polyfit(q.index,q[metric],1)[0]
                rr.append(dict(split=split,variant=variant,model_seed=seed,kind=kind,metric=metric,
                               aurc=float(np.trapezoid(ratio.values,q.index)),slope=float(slope),
                               clean=float(baseline),at_full_missing=float(q.loc[1.,metric])))
    pd.DataFrame(rr).to_csv(OUT/'robustness_aurc.csv',index=False)
    # Summary dispersions across five model seeds; placement is averaged within each seed first.
    rows=[]
    for (split,variant,seed),part in allrows.groupby(['split','variant','model_seed']):
        clean=part[part.family=='clean'].iloc[0]
        missing=part[(part.family=='rate')&(part.rate.isin([.2,.4,.6]))]
        row=dict(split=split,variant=variant,model_seed=seed)
        for k in ['accuracy','macro_f1','weighted_f1','mae','corr']:
            row['clean_'+k]=float(clean[k]);row['missing_'+k]=float(missing[k].mean())
        rows.append(row)
    seed_summary=pd.DataFrame(rows);seed_summary.to_csv(OUT/'model_summary_per_seed.csv',index=False)
    agg=seed_summary.groupby(['split','variant']).agg({k:['mean','std'] for k in seed_summary if k not in ['split','variant','model_seed']})
    agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(OUT/'model_summary.csv',index=False)


def grid_plots(df):
    main=df[(df.variant=='A0')&(df.stratum=='all')&(df.split=='valid')]
    clean=main[main.family=='clean'];rate=main[main.family=='rate']
    fig,axs=plt.subplots(2,4,figsize=(12,5.5),sharex=True)
    for j,kind in enumerate(TYPES):
        ax=axs.flat[j];sub=rate[rate.kind==kind]
        for metric,color,label in [('macro_f1',COLORS[0],'Macro-F1'),('mae',COLORS[1],'MAE')]:
            values=sub.groupby(['model_seed','rate'])[metric].mean().unstack()
            values[0]=clean.set_index('model_seed')[metric];values=values.sort_index(axis=1)
            mean=values.mean(0);std=values.std(0)
            ax.plot(mean.index,mean.values,color=color,label=label)
            ax.fill_between(mean.index,(mean-std).values,(mean+std).values,color=color,alpha=.30)
        ax.set_title(kind);ax.set_xlabel('Missing fraction');ax.set_ylim(0,1.3)
    axs.flat[-1].axis('off');axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(),loc='center')
    fig.tight_layout();save(fig,'performance_missing_rate')
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for j,r in enumerate([.3,.5]):
        pivot=main[(main.family=='position')&(main.rate==r)].pivot_table(index='kind',columns='location',values='mae').reindex(TYPES)
        im=axs[j].imshow(pivot,cmap=SEQUENTIAL,aspect='auto');axs[j].set_xticks(range(3),['Start','Middle','End'])
        axs[j].set_yticks(range(7),TYPES);axs[j].set_title(f'MAE, missing fraction {r}')
        for y in range(7):
            for x in range(3):axs[j].text(x,y,f'{pivot.iloc[y,x]:.3f}',ha='center',va='center',fontsize=8,color=heatmap_text_color(im,pivot.iloc[y,x]))
        fig.colorbar(im,ax=axs[j],shrink=.85)
    fig.tight_layout();save(fig,'position_heatmap')
    fig,axs=plt.subplots(1,2,figsize=(10,3.2))
    frag=main[main.family=='fragment'].groupby(['kind','segments']).mae.mean().unstack().reindex(TYPES)
    frag.plot.bar(ax=axs[0],color=COLORS[:3],rot=0);axs[0].set_ylabel('MAE');axs[0].set_xlabel('Missing modalities');axs[0].legend(title='Segments')
    col=main[main.family=='colocated'].groupby(['kind','colocated']).mae.mean().unstack().reindex(['TA','TV','AV','TAV'])
    col.plot.bar(ax=axs[1],color=COLORS[:2],rot=0);axs[1].set_ylabel('MAE');axs[1].set_xlabel('Missing modalities');axs[1].legend(['Independent','Co-located'])
    fig.tight_layout();save(fig,'fragment_colocation')
    fig,axs=plt.subplots(1,2,figsize=(10,3.2))
    for m,(kind,color) in enumerate(zip(['T','A','V'],COLORS)):
        q=rate[rate.kind==kind].groupby('rate')[[f'w_{kind.lower()}',f'branch_mae_{kind.lower()}']].mean()
        axs[0].plot(q.index,q.iloc[:,0],label=kind,color=color,marker='o')
        axs[1].plot(q.index,q.iloc[:,1],label=kind,color=color,marker='o')
    for ax in axs:ax.set_xlabel('Missing fraction');ax.legend()
    axs[0].set_ylabel('Gate weight');axs[1].set_ylabel('Branch MAE (present samples)')
    fig.tight_layout();save(fig,'gate_branch_error')
    fig,axs=plt.subplots(1,3,figsize=(12,3.1))
    for k,(kind,color) in enumerate(zip(TYPES,COLORS)):
        q=main[(main.family=='duration')&(main.kind==kind)].groupby('count').mae.mean()
        axs[0].plot(q.index,q.values,label=kind,color=color,marker=['o','s','D','P','v','X','h'][k],linestyle=['-','--','-.',':','-','--','-.'][k])
        s=main[(main.family=='scan')&(main.kind==kind)].groupby('location').mae.mean()
        axs[2].plot(s.index,s.values,label=kind,color=color,linestyle=['-','--','-.',':','-','--','-.'][k])
    for m,color in zip('tav',COLORS):
        sub=main[(main.family=='duration')&(main.kind==m.upper())]
        q=sub.groupby('count')[f'recon_r2_{m}'].mean()
        axs[1].plot(q.index,q.values,label=m.upper(),color=color,marker='o')
    axs[0].set_xlabel('Removed positions');axs[0].set_ylabel('MAE');axs[0].legend(ncol=2,fontsize=7)
    axs[1].set_xlabel('Removed positions');axs[1].set_ylabel('Reconstruction pooled R2');axs[1].legend()
    axs[2].set_xlabel('Window relative start');axs[2].set_ylabel('MAE, 3-position window')
    fig.tight_layout();save(fig,'duration_reconstruction_scan')
    fig,axs=plt.subplots(1,2,figsize=(11,3))
    summary=pd.read_csv(OUT/'model_summary.csv');q=summary[(summary.split=='valid')&summary.variant.str.startswith('A')].sort_values('variant')
    x=np.arange(len(q));axs[0].bar(x,q.missing_macro_f1_mean,yerr=q.missing_macro_f1_std,color=[COLORS[0] if v=='A0' else '#A6A6A6' for v in q.variant],capsize=2)
    axs[1].bar(x,q.missing_mae_mean,yerr=q.missing_mae_std,color=[COLORS[1] if v=='A0' else '#A6A6A6' for v in q.variant],capsize=2)
    for ax in axs:ax.set_xticks(x,q.variant);ax.set_xlabel('Ablation')
    axs[0].set_ylabel('Macro-F1');axs[1].set_ylabel('MAE');fig.tight_layout();save(fig,'ablation')


def confusion_matrix_plot():
    metrics=json.loads((OUT/'ensemble_metrics.json').read_text())
    fig,axs=plt.subplots(1,4,figsize=(8.4,2.4))
    for row,split in enumerate(['valid','test']):
        for col,condition in enumerate(['clean','missing']):
            ax=axs[2*row+col];cm=np.asarray(metrics[split][condition]['confusion'])
            im=ax.imshow(cm,cmap=SEQUENTIAL);ax.set_title(f'{split}, {condition}')
            ax.set_xticks(range(3),['Neg','Neu','Pos']);ax.set_yticks(range(3),['Neg','Neu','Pos'])
            ax.set_xlabel('Predicted');ax.set_ylabel('True')
            for i in range(3):
                for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,color=heatmap_text_color(im,cm[i,j]))
    fig.tight_layout(pad=.7,w_pad=1.0);save(fig,'confusion_matrices')


def diagnostic_plots():
    confusion_matrix_plot()
    clean=np.load(OUT/'valid_ensemble_clean.npz');missing=np.load(OUT/'valid_ensemble_missing.npz');d=load_split('valid')
    if '--figures-only' in sys.argv:
        xy=np.load(OUT/'tsne_coordinates.npz')['xy']
    else:
        joined=np.concatenate([clean['z'],missing['z']]);pca=PCA(n_components=30,random_state=337).fit_transform(joined)
        xy=TSNE(n_components=2,perplexity=30,random_state=337,init='pca',learning_rate='auto',max_iter=1000).fit_transform(pca)
        np.savez_compressed(OUT/'tsne_coordinates.npz',xy=xy)
    fig,ax=plt.subplots(figsize=(6,4))
    for cls,label in zip([-1,0,1],['Negative','Neutral','Positive']):
        color=POLARITY[label]
        sel=np.sign(d['y'])==cls
        ax.scatter(xy[:len(sel)][sel,0],xy[:len(sel)][sel,1],s=9,c=color,label=f'{label}: clean',alpha=1)
        ax.scatter(xy[len(sel):][sel,0],xy[len(sel):][sel,1],s=13,facecolors='none',edgecolors=color,label=f'{label}: missing',alpha=1,linewidths=.8)
    ax.set_xticks([]);ax.set_yticks([]);ax.legend(ncol=2,fontsize=8);fig.tight_layout();save(fig,'tsne_clean_missing')
    errors=pd.read_csv(OUT/'stratified_errors.csv');q=errors[(errors.split=='valid')&errors.group.str.startswith('intensity_')]
    pivot=q.pivot(index='group',columns='condition',values='mae')
    fig,ax=plt.subplots(figsize=(7,3));pivot.plot.bar(ax=ax,color=COLORS[:2],rot=0)
    ax.set_xticklabels([s.replace('intensity_','') for s in pivot.index]);ax.set_xlabel('True intensity group');ax.set_ylabel('MAE')
    fig.tight_layout();save(fig,'stratified_errors')


def special_plots():
    audit=json.loads((OUT/'data_audit.json').read_text())['special']['attachment3']
    pred=pd.read_csv(OUT/'attachment3_predictions.csv');fig,axs=plt.subplots(1,3,figsize=(11,8),sharey=True)
    for m,mod in enumerate(MODALITIES):
        for i,r in enumerate(audit):
            L=r['valid_len'];axs[m].barh(i,L,height=.7,color='#E6E6E6')
            for s,e in r[f'miss_{mod}_intervals']:
                axs[m].barh(i,e-s,left=s,height=.7,color=COLORS[m])
        axs[m].set_title(mod);axs[m].set_xlabel('Aligned position');axs[m].set_xlim(0,50)
    axs[0].set_yticks(range(30),[str(i+1).zfill(2) for i in range(30)]);axs[0].invert_yaxis()
    axs[0].legend(handles=[Patch(facecolor='#E6E6E6',label='Valid range'),Patch(facecolor=COLORS[6],label='Unobserved interval (modality color)')])
    fig.tight_layout();save(fig,'attachment3_missing_gantt')
    fig,axs=plt.subplots(1,2,figsize=(10,3.3))
    x=np.arange(1,31);colors=[POLARITY[v] for v in pred.pred_polarity]
    axs[0].bar(x,pred.pred_intensity,color=colors,yerr=pred.ensemble_std,capsize=2);axs[0].axhline(0,c='#555555',lw=.6)
    axs[0].set_xlabel('Attachment 3 sample');axs[0].set_ylabel('Intensity (error bar: seed SD)')
    axs[0].legend(handles=[Patch(facecolor=c,label=label) for label,c in POLARITY.items()])
    for m,mod in enumerate(MODALITIES):axs[1].scatter(pred[f'miss_ratio_{"tav"[m]}'],pred[f'w_{mod}'],c=COLORS[m],label=mod,s=20,alpha=1)
    axs[1].set_xlabel('Observed zero fraction');axs[1].set_ylabel('Gate weight');axs[1].legend()
    fig.tight_layout();save(fig,'attachment3_predictions_gates')


def reconstruction_figure():
    import torch
    from model import RMNet
    from train import device_data
    from missing import fixed_mask
    torch.set_num_threads(2)
    ck=torch.load(ROOT/'models/rmnet/weights_fp16.pt',map_location='cpu',weights_only=False)
    model=RMNet(ck['config'],ck['variant']).eval();model.load_state_dict(ck['state_dict'])
    d=load_split('valid');idx=int(np.flatnonzero((d['lengths']>=25)&(d['lengths']<=30))[0]);L=d['lengths'][idx]
    sub={k:v[idx:idx+1] for k,v in d.items()};t=device_data(sub,'cpu');keep=fixed_mask(sub['lengths'],'TAV',.4,331)
    with torch.inference_mode():p=model(*t,mask=torch.as_tensor(keep))
    fig,axs=plt.subplots(3,3,figsize=(11,5.6))
    for m,mod in enumerate(MODALITIES):
        original=sub[mod][0,:L,:16];damaged=original*keep[0,m,:L,None]
        filled=np.where((~keep[0,m,:L,None])&sub['obs'][0,m,:L,None],p['recon'][m][0,:L,:16].numpy(),original)
        for j,mat in enumerate([original,damaged,filled]):
            axs[m,j].imshow(mat.T,aspect='auto',cmap=DIVERGING,vmin=-3,vmax=3)
            axs[m,j].set_title(f'{mod}: {["original","masked","reconstructed"][j]}',fontsize=9)
            axs[m,j].set_ylabel('First 16 features');axs[m,j].set_xlabel('Position')
    fig.tight_layout();save(fig,'reconstruction_heatmap')
    if '--figures-only' not in sys.argv:
        write_json(OUT/'reconstruction_example.json',{'valid_index':idx,'sample_id':str(d['ids'][idx]),'mask_seed':331,'features':'first 16 normalized dimensions per modality'})


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    apply_style()
    if '--figures-only' in sys.argv:
        df=pd.read_csv(OUT/'missing_grid_raw.csv',low_memory=False)
    else:
        df=load_results();summary_tables(df)
    grid_plots(df);diagnostic_plots();special_plots();reconstruction_figure()
    fidelity=OUT/'fidelity_curves.csv'
    if fidelity.exists():
        d=pd.read_csv(fidelity);fig,axs=plt.subplots(1,2,figsize=(9,3))
        for mode,color in zip(['importance','random'],COLORS):
            for j,op in enumerate(['delete','retain']):
                q=d[(d.ordering==mode)&(d.operation==op)];axs[j].plot(q.fraction,q['corr'],label=mode,color=color,marker='o')
                axs[j].set_title(op);axs[j].set_xlabel('Fraction of observed positions');axs[j].set_ylabel('Pearson r');axs[j].legend()
        fig.tight_layout();save(fig,'fidelity_curves')
    if '--figures-only' in sys.argv:
        return
    # Derive a descriptive token-duration scale from P1 measured word alignments only.
    durations=[];word_durations=[];source_files=[]
    for path in sorted((ROOT/'outputs/problem1/word_times').glob('*.json')):
        source_files.append(str(path.relative_to(ROOT)))
        for word in json.loads(path.read_text()):
            n=len(word.get('bert_subtokens',[]));dt=word['e']-word['s']
            if word.get('timing_usable') and n>0 and dt>0:
                word_durations.append(dt);durations.extend([dt/n]*n)
    if durations:write_json(OUT/'duration_time_scale.json',{'n_words':len(word_durations),'n_subtokens':len(durations),
        'word_duration_median_s':float(np.median(word_durations)),'subtoken_duration_median_s':float(np.median(durations)),
        'subtoken_quantiles_s':np.quantile(durations,[.25,.5,.75]).tolist(),
        'approx_seconds_by_k':{str(k):float(k*np.median(durations)) for k in [1,2,3,5,8,12]},
        'scope':'Illustrative P1 scale only; not measured times for attachment2/3, excludes inter-word pauses.',
        'sources':source_files})
    print('Saved tables and publication PDF/PNG figures.')


if __name__=='__main__':main()
