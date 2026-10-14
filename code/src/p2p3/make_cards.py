"""Deterministic case selection, Markdown evidence cards, and quantitative figures."""
import json
import sys
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from project_paths import resolve_asset
from data import ROOT,MODALITIES,write_json
from p3_common import OUT3,LABELS
from cards_markdown import markdown_text, markdown_table, sample_link, write_index

sys.path.insert(0,str(ROOT/'src/common'))
from plot_style import (COLORS, INTERVAL_COLORS, SEQUENTIAL, INK, apply_style, boxplot,
                         heatmap_text_color, save_figure)
FIG=OUT3/'figures'


def save(fig,name):
    name=name.replace('$','')
    save_figure(fig,FIG,name)


def choose(split):
    samples=[json.loads(p.read_text()) for p in sorted((OUT3/'samples'/split).glob('*.json'))]
    chosen=[]
    for label in LABELS:
        candidates=[s for s in samples if s['prediction']['pred_polarity']==label and s['prediction']['agree_flag']]
        if split=='valid':candidates=[s for s in candidates if LABELS[int(np.sign(s['truth'])+1)]==label]
        if not candidates:continue
        median=np.median([s['prediction']['seed_sd'] for s in candidates])
        low=[s for s in candidates if s['prediction']['seed_sd']<=median]
        best=max(low,key=lambda s:s['prediction']['margin'])
        chosen.append((f'typical_{label}',best))
    if split=='valid':
        for label in LABELS:
            errors=[s for s in samples if s['prediction']['pred_polarity']==label and LABELS[int(np.sign(s['truth'])+1)]!=label]
            if errors:chosen.append((f'error_{label}',max(errors,key=lambda s:s['prediction']['margin'])))
    else:
        disagreements=[s for s in samples if not s['prediction']['agree_flag']]
        if disagreements:chosen.append(('modality_disagreement',max(disagreements,key=lambda s:s['prediction']['margin'])))
        low=[s for s in samples if 'sample_WER' in s['prediction']['time_quality_flags']]
        if low:chosen.append(('low_trust_timing',max(low,key=lambda s:s['prediction']['WER'])))
    seen=set();result=[]
    for role,s in chosen:
        if s['prediction']['id'] not in seen:result.append((role,s));seen.add(s['prediction']['id'])
    return result


def frames(sample):
    import av
    desired={r['keyframe']['frame_index'] for r in sample['top_evidence']['vision'] if r.get('keyframe')}
    images={}
    if not desired:return images
    with av.open(str(resolve_asset(sample['video']))) as c:
        for i,frame in enumerate(c.decode(video=0)):
            if i in desired:images[i]=frame.to_image()
            if i>=max(desired):break
    return images


def card_figure(split,s):
    p=s['prediction'];sid=p['id'];color=COLORS[0]
    if split=='valid':
        fig,axs=plt.subplots(3,1,figsize=(10,5),sharex=True)
        for m,ax in enumerate(axs):
            pp=s['positions'][m];x=[r['position'] for r in pp];val=[r['score'] if r['eligible'] else 0 for r in pp]
            ax.bar(x,val,color=COLORS[m],width=.85);ax.set_ylabel(MODALITIES[m])
            ax.margins(y=.45)
            evidence={}
            for r in s['top_evidence'][MODALITIES[m]]:
                evidence.setdefault(r['word'],[]).append(r['position'])
                ax.scatter(r['position'],r['score'],s=18,color=COLORS[m],edgecolors='white',linewidths=.6,zorder=4)
            for j,(word,positions) in enumerate(evidence.items()):
                ax.text(.02+j*.33,.95,word+' | pos '+', '.join(map(str,positions)),
                    transform=ax.transAxes,ha='left',va='top',fontsize=7,color=INK,
                    bbox={'boxstyle':'square,pad=.25','fc':'white','ec':'#808080','lw':.5})
        axs[-1].set_xlabel('Official aligned WordPiece position (no measured validation times)')
        fig.suptitle(f'{sid} | y={p["pred_intensity"]:.3f} | true={s["truth"]:.3f}')
        fig.tight_layout();save(fig,f'card_{split}_{sid}');return
    aligned=json.loads((OUT3/'alignment'/f'{sid}.json').read_text());a=np.load(OUT3/'alignment'/f'{sid}.npz')
    image_map=frames(s)
    fig=plt.figure(figsize=(11,8.5));gs=fig.add_gridspec(4,3,height_ratios=[1.25,1,1,1.8],hspace=.95)
    ax=fig.add_subplot(gs[0,:]);maxscore=max([r['score'] for r in s['positions'][0]]) or 1
    # Word-level maximum combines duplicate WordPieces for readable highlighting.
    for j,w in enumerate(aligned['words']):
        related=[r for r in s['positions'][0] if r.get('word_char_start')==w['char_start']]
        val=max([r['score'] for r in related],default=0)/maxscore
        ax.add_patch(Rectangle((w['s'],0),w['e']-w['s'],1,facecolor=SEQUENTIAL(.10+.8*val),
                               hatch=None if w['timing_usable'] else '///',edgecolor='black',linewidth=.5))
        ax.text((w['s']+w['e'])/2,1.04+(j%2)*.28,w['word'],rotation=35,ha='left',fontsize=6)
    duration=aligned['stream']['duration'];ax.set_xlim(0,duration);ax.set_ylim(0,2);ax.set_yticks([]);ax.set_ylabel('Text')
    ax=fig.add_subplot(gs[1,:]);ax.plot(a['audio_times'],np.where(a['f0']>0,a['f0'],np.nan),color=COLORS[1],lw=1,label='F0')
    ax.set_ylabel('F0 (Hz)');ax.set_xlim(0,duration)
    twin=ax.twinx();twin.plot(a['audio_times'],a['rms'],color=COLORS[6],lw=.9,label='RMS');twin.set_ylabel('RMS')
    for r in s['top_evidence']['audio']:
        if r.get('time_start') is not None:
            ax.axvspan(r['time_start'],r['time_end'],facecolor=INTERVAL_COLORS[1],
                       edgecolor=COLORS[1],linewidth=.7,zorder=.5)
    ax=fig.add_subplot(gs[2,:]);pp=s['positions'][0]
    valid=[r for r in pp if r.get('time_start') is not None and r['eligible']]
    ax.bar([(r['time_start']+r['time_end'])/2 for r in valid],
           [abs(s['joint_occlusion'][r['position']]) for r in valid],
           width=[max(.015,(r['time_end']-r['time_start'])*.8) for r in valid],color=COLORS[2])
    ax.set_xlim(0,duration);ax.set_ylabel('Joint |delta y|');ax.set_xlabel('Seconds relative to common audio/video start')
    for j in range(3):
        ax=fig.add_subplot(gs[3,j]);ax.axis('off')
        if j>=len(s['top_evidence']['vision']):continue
        r=s['top_evidence']['vision'][j];f=r.get('keyframe')
        if f and f['frame_index'] in image_map:
            ax.imshow(image_map[f['frame_index']]);ax.set_title(f"{textwrap.shorten(r['word'],width=20)} | frame {f['frame_index']}\nPTS {f['pts_relative']:.3f}s | face={f['face']} | q={r['timing_usable']}",fontsize=8)
        else:ax.text(.5,.5,'No observed visual evidence',ha='center')
    fig.suptitle(f"Attachment4 {sid} | {p['pred_polarity']} ({p['pred_intensity']:.3f}) | main={p['main_modality_shap']}\nWER={p['WER']:.2f}, CTC={p['CTC_mean']:.2f}; hatching marks candidate low-trust timing",fontsize=11)
    save(fig,f'card_{split}_{sid}')


def markdown_card(role,s,split):
    p=s['prediction'];sid=p['id']
    modal=pd.DataFrame(s['modalities']).drop(columns=['mass_share_note']).round(5)
    tables=[]
    for m in MODALITIES:
        rows=[]
        for r in s['top_evidence'][m]:
            f=r.get('keyframe') or {}
            rows.append(dict(position=r['position'],word=r['word'],chars=f"[{r['char_start']},{r['char_end']})",
                seconds=f"[{r.get('time_start')},{r.get('time_end')})" if r.get('time_start') is not None else 'relative position only',
                O=round(r['occlusion_delta'],5),G=round(r['gradient_input'],5),rank=round(r['rank_fusion'],4),
                score=round(r['score'],5),flip=r['polarity_flip'],frame=f.get('frame_index'),face=f.get('face'),q=r.get('timing_usable')))
        frame=pd.DataFrame(rows)
        if rows:
            tables.extend([f'### {m} top-3',markdown_table(frame.columns,frame.itertuples(index=False,name=None))])
    truth=f"，真值 {s['truth']:.3f}" if 'truth' in s else ''
    return '\n\n'.join([
        f"## {markdown_text(split)} / {markdown_text(sid)} · {markdown_text(role)}",
        '> '+markdown_text(p['raw_text']),
        f"强度 {p['pred_intensity']:.4f}，{p['pred_polarity']}{truth}；种子SD {p['seed_sd']:.4f}；margin {p['margin']:.4f}。"
        f"主要参考模态 {p['main_modality_shap']}；内部贡献一致 {p['agree_flag']}；并列标志 {p['tie_flag']}。",
        f"有序概率 {p['ordinal_p_neg']:.3f} / {p['ordinal_p_neu']:.3f} / {p['ordinal_p_pos']:.3f}；"
        f"辅助概率 {p['aux_p_neg']:.3f} / {p['aux_p_neu']:.3f} / {p['aux_p_pos']:.3f}（均为描述量）。"
        f"κ={p['flip_k']}，删失={p['flip_censored']}，随机κ均值={p['flip_k_rand']:.2f}。",
        markdown_table(modal.columns,modal.itertuples(index=False,name=None)),
        f"![共享时间轴或相对位置解释图](../figures/card_{split}_{sid.replace('$','')}.png)",
        *tables,
        markdown_text(s['alignment_notice'])+' '+markdown_text(p['notes']),
        f'[全部位置分数 JSON]({sample_link(split,sid)})',
    ])


def figures():
    df=pd.read_csv(OUT3/'validation/sample_diagnostics.csv');rule=json.loads((OUT3/'explanation_rules.json').read_text())
    summary=json.loads((OUT3/'validation/summary.json').read_text())
    curve=pd.read_csv(OUT3/'validation/fidelity_curves.csv')
    fig,axs=plt.subplots(2,3,figsize=(10,5.5))
    for row,operation in enumerate(['delete','insert']):
        for col,metric in enumerate(['abs_delta','mae','pearson']):
            ax=axs[row,col]
            for i,method in enumerate(['occlusion','gradient','alpha','fusion','random']):
                sub=curve[(curve.scope=='joint')&(curve.method==method)&(curve.operation==operation)&(curve.width==(0 if method=='random' else rule['width']))]
                ax.plot(sub.fraction,sub[metric],label=method,color=COLORS[6] if method=='random' else COLORS[i],marker=['o','s','^','D','x'][i],markersize=3)
            ax.set_title(f'{operation}: {metric}');ax.set_xlabel('Fraction of observed content positions')
    axs[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'fidelity_curves')
    fig,axs=plt.subplots(2,3,figsize=(10,5.8))
    for j,m in enumerate(MODALITIES):
        # Raw internal quantities use the full explanation dump.
        from fidelity_eval import load_e
        e,_=load_e('valid',rule['width'])
        axs[0,j].scatter(e['contributions'][:,j],e['shapley'][:,j],s=7,alpha=1,color=COLORS[j],linewidths=0)
        axs[0,j].set_xlabel('Internal contribution');axs[0,j].set_ylabel('Exact Shapley');axs[0,j].set_title(m)
        axs[1,j].scatter(e['w'][:,j],df['share_'+m],s=7,alpha=1,color=COLORS[j],linewidths=0)
        axs[1,j].set_xlabel('Reliability gate');axs[1,j].set_ylabel('Absolute Shapley share')
    fig.tight_layout();save(fig,'modality_consistency')
    fig,axs=plt.subplots(1,3,figsize=(10,3))
    cm=np.array(summary['validation']['confusion']);im=axs[0].imshow(cm,cmap=SEQUENTIAL)
    for i in range(3):
        for j in range(3):axs[0].text(j,i,cm[i,j],ha='center',va='center',color=heatmap_text_color(im,cm[i,j]))
    axs[0].set_xticks(range(3),['Neg','Neu','Pos']);axs[0].set_yticks(range(3),['Neg','Neu','Pos']);axs[0].set_xlabel('Prediction');axs[0].set_ylabel('Truth')
    counts=[summary['main_counts'][m] for m in MODALITIES];axs[1].bar(MODALITIES,counts,color=COLORS[:3]);axs[1].set_ylabel('Main-modality count')
    for col,label in [(False,'Correct'),(True,'Error')]:
        axs[2].hist(df.loc[df.error==col,'margin'],bins=16,histtype='step',
                    linewidth=1.2,color=COLORS[int(col)],label=label,density=True,
                    joinstyle='miter')
    axs[2].set_xlabel('Threshold margin');axs[2].legend();fig.tight_layout();save(fig,'performance_errors')
    fig,axs=plt.subplots(2,3,figsize=(10,5.5))
    for j,m in enumerate(MODALITIES):
        vals=[df.loc[df.true_class==c,'shap_'+m] for c in range(3)]
        boxplot(axs[0,j],vals,['Neg','Neu','Pos'],[COLORS[j]]);axs[0,j].set_title(m);axs[0,j].set_ylabel('Shapley by true polarity')
        bins=pd.cut(df.intensity,[-.01,.5,1,2,3],include_lowest=True)
        vals=[g['share_'+m] for _,g in df.groupby(bins,observed=True)]
        boxplot(axs[1,j],vals,['0-.5','.5-1','1-2','2-3'],[COLORS[j]]);axs[1,j].set_ylabel('Share by |true intensity|')
    fig.tight_layout();save(fig,'modality_strata')
    hist=np.zeros((3,3),int)
    for path in (OUT3/'samples/valid').glob('*.json'):
        s=json.loads(path.read_text());m=MODALITIES.index(s['prediction']['main_modality_shap']);length=s['prediction']['valid_len']-2
        for r in s['top_evidence'][MODALITIES[m]]:hist[m,min(2,int((r['position']-1)/max(length,1)*3))]+=1
    if '--figures-only' not in sys.argv:
        pd.DataFrame(hist,index=MODALITIES,columns=['early','middle','late']).to_csv(OUT3/'validation/important_position_thirds.csv')
    fig,ax=plt.subplots(figsize=(5,3));bottom=np.zeros(3)
    for j,label in enumerate(['early','middle','late']):
        v=hist[:,j]/np.maximum(hist.sum(1),1);ax.bar(MODALITIES,v,bottom=bottom,label=label,color=COLORS[j]);bottom+=v
    ax.set_ylabel('Top-3 position fraction\nwithin main modality');ax.set_ylim(0,1.04);ax.legend();fig.tight_layout();save(fig,'important_positions')


def main():
    from p3_common import require_legacy_protocol
    require_legacy_protocol()
    FIG.mkdir(exist_ok=True);(OUT3/'cards').mkdir(exist_ok=True)
    apply_style()
    figures()
    if '--figures-only' in sys.argv:
        selection=json.loads((OUT3/'cards/selection.json').read_text())['samples']
        for selected in selection:
            split=selected['split'];sid=selected['id']
            sample=json.loads((OUT3/'samples'/split/f'{sid}.json').read_text())
            card_figure(split,sample)
        print('Redrew existing quantitative figures and',len(selection),'selected cards',flush=True)
        return
    cards=[];manifest=[]
    for split in ['valid','attachment4']:
        for role,s in choose(split):
            card_figure(split,s);cards.append(markdown_card(role,s,split))
            manifest.append(dict(split=split,role=role,id=s['prediction']['id']))
    write_json(OUT3/'cards/selection.json',dict(rule='For each class: correct (validation only), contribution/Shapley agree, seed SD at or below candidate median, greatest margin; error cases greatest margin per predicted class; attachment4 disagreement and greatest WER.',samples=manifest))
    write_index(OUT3,cards,title='问题3 · 可复核的多模态解释卡',
                introduction='时间为独立CTC估计，低可信候选显式标记；注意力与门控仅作辅助；证据不等于人类情感的因果解释。')
    print('Created',len(cards),'cards and quantitative figures',flush=True)


if __name__=='__main__':main()
