"""Aggregate the remaining baseline tables and measurable hypothesis checks."""
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT,OUT,write_json
from figures import COLORS,save
from plot_style import apply_style


def main():
    apply_style()
    if '--figures-only' in sys.argv:
        df=pd.read_csv(OUT/'missing_grid_raw.csv',low_memory=False)
        sm=pd.read_csv(OUT/'imputation_summary.csv').set_index(['split','method'])
        plot_tables(df,sm)
        return
    files=sorted((OUT/'analysis/imputation').glob('imputation_*_s*.csv'))
    if len(files)!=10:raise RuntimeError(f'Expected 10 imputation jobs; got {len(files)}')
    imp=pd.concat([pd.read_csv(f) for f in files]);imp.to_csv(OUT/'imputation_raw.csv',index=False)
    sm=imp[imp.rate.isin([.2,.4,.6])].groupby(['split','method','seed'])[['mae','macro_f1']].mean().groupby(['split','method']).agg(['mean','std'])
    sm.columns=['_'.join(c) for c in sm.columns];sm.reset_index().to_csv(OUT/'imputation_summary.csv',index=False)
    df=pd.read_csv(OUT/'missing_grid_raw.csv',low_memory=False)
    main=df[(df.variant=='A0')&(df.split=='valid')&(df.stratum=='all')]
    rate=main[main.family=='rate'];records=[]
    for mod in 'TAV':
        q=rate[(rate.kind==mod)&(rate.rate<1)].groupby('rate').mean(numeric_only=True)
        records.append({'hypothesis':'gate_decreases_with_partial_missing','modality':mod,
                        'spearman_rate_weight':float(spearmanr(q.index,q[f'w_{mod.lower()}']).statistic),
                        'spearman_gate_branch_mae':float(spearmanr(q[f'w_{mod.lower()}'],q[f'branch_mae_{mod.lower()}']).statistic)})
    clean=float(main[main.family=='clean'].mae.mean());table=rate.groupby(['kind','rate']).mae.mean()
    for pair in ['TA','TV','AV']:
        for r in [.2,.4,.6,.8,1.]:
            excess=float(table.loc[pair,r]-table.loc[pair[0],r]-table.loc[pair[1],r]+clean)
            records.append({'hypothesis':'superadditive_pair_error','kind':pair,'rate':r,'excess_mae':excess})
    write_json(OUT/'hypothesis_checks.json',records)
    duration=df[(df.variant=='A0')&(df.split=='valid')&(df.family=='duration')]
    duration.groupby(['kind','count','stratum'])[['mae','macro_f1']].agg(['mean','std']).to_csv(OUT/'duration_by_length.csv')
    plot_tables(df,sm)
    totals={'trained_models':len(list((OUT/'runs').glob('*/complete.json'))),
            'grid_jobs':len(list((OUT/'evaluation').glob('*/*_complete.json'))),
            'grid_rows_including_strata':len(df),'grid_rows_all_samples':int((df.stratum=='all').sum()),
            'imputation_rows':len(imp),'special_predictions':len(pd.read_csv(OUT/'attachment3_predictions.csv'))}
    write_json(OUT/'experiment_totals.json',totals)
    print(totals)


def retention_curve(subset,metric):
    """Average per-model, per-type ratios, matching robustness_aurc.csv."""
    curves=[]
    for _,part in subset.groupby('model_seed'):
        base=float(part[part.family=='clean'].iloc[0][metric])
        for _,group in part[part.family=='rate'].groupby('kind'):
            q=group.groupby('rate')[metric].mean()
            q.loc[0.]=base;q=q.sort_index()
            curves.append(base/q.clip(lower=1e-8) if metric=='mae' else q/max(base,1e-8))
    if not curves:
        raise ValueError('No per-model retention curves')
    return pd.concat(curves,axis=1).mean(axis=1)


def plot_tables(df,sm):
    fig,axs=plt.subplots(1,2,figsize=(9,3.1))
    for k,(variant,color) in enumerate(zip(['A0','A1','A4','A7','B_T'],COLORS)):
        subset=df[(df.variant==variant)&(df.split=='valid')&(df.stratum=='all')]
        for j,metric in enumerate(['macro_f1','mae']):
            rr=retention_curve(subset,metric)
            axs[j].plot(rr.index,rr.values,label=variant,color=color,linestyle=['-','--','-.',':','--'][k],marker=['o','s','D','P','v'][k],markevery=2)
            axs[j].set_xlabel('Missing fraction');axs[j].set_ylabel(f'{metric} retention');axs[j].legend(fontsize=7)
    fig.tight_layout();save(fig,'robustness_retention')
    fig,axs=plt.subplots(1,2,figsize=(9,3.1));q=sm.reset_index();q=q[q.split=='valid']
    for j,metric in enumerate(['macro_f1','mae']):
        axs[j].bar(q.method,q[f'{metric}_mean'],yerr=q[f'{metric}_std'],color=COLORS[:4],capsize=2)
        axs[j].set_ylabel(metric);axs[j].set_xlabel('Imputation into clean-trained concatenation model')
    fig.tight_layout();save(fig,'imputation_baselines')


if __name__=='__main__':main()
