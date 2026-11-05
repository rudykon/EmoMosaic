"""Publication figures and evidence cards from final-fusion artifacts only."""
import json
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT, MODALITIES, write_json
from p3_common import OUT3
from layered_p3_cards import render_card
from cards_markdown import fusion_card, write_index, append_completion

sys.path.insert(0,str(ROOT/'src/common'))
from plot_style import apply_style, save_figure
COLORS=['#0072B2','#D55E00','#009E73']
MODALITY_NAMES={'text':'文本','audio':'语音','vision':'视觉'}
FIG=OUT3/'figures'


def save(fig,name):
    save_figure(fig,FIG,name)
    plt.close(fig)


def choose(split):
    samples=[json.loads(p.read_text()) for p in sorted((OUT3/'samples'/split).glob('*.json'))]
    selected=[];seen=set()
    def add(role,candidates):
        candidates=[s for s in candidates if s['prediction']['id'] not in seen]
        if candidates:
            s=max(candidates,key=lambda s:s['prediction']['probability_margin'])
            selected.append((role,s));seen.add(s['prediction']['id'])
    for c,label in enumerate(['Negative','Neutral','Positive']):
        candidates=[s for s in samples if s['prediction']['pred_polarity']==label]
        if split=='valid':candidates=[s for s in candidates if int(np.sign(s['truth'])+1)==c]
        add('典型预测 '+label,candidates)
    if split=='valid':
        for c,label in enumerate(['Negative','Neutral','Positive']):
            add('误判案例 '+label,[s for s in samples if s['prediction']['pred_polarity']==label and int(np.sign(s['truth'])+1)!=c])
    else:
        add('强度与分类主模态不同',[s for s in samples if s['prediction']['main_modality_intensity']!=s['prediction']['main_modality_classification']])
        ranked=sorted(samples,key=lambda s:s['prediction']['WER'],reverse=True)
        add('时间映射低可信',[s for s in ranked if s['prediction']['WER']==ranked[0]['prediction']['WER']])
    return selected


def card(split,role,s):
    sid=s['prediction']['id'];name=f'card_{split}_{sid}'.replace('$','')
    render_card(split,s)
    return fusion_card(split,role,s,name),dict(split=split,id=sid,role=role,figure=name)


def main():
    FIG.mkdir(exist_ok=True);(OUT3/'cards').mkdir(exist_ok=True);apply_style()
    curves=pd.read_csv(OUT3/'validation/fidelity_curves.csv')
    fig,axs=plt.subplots(2,2,figsize=(8,5.6))
    for j,target in enumerate(['intensity','classification']):
        for k,op in enumerate(['delete','retain']):
            r=curves[(curves.target==target)&(curves.operation==op)]
            ax=axs[j,k];ax.plot(r.fraction,r.selected,'o-',color=COLORS[0],label='Output occlusion')
            ax.plot(r.fraction,r.random,'s--',color=COLORS[1],label='Random (5 repeats)')
            ax.set_xlabel('Fraction of content positions');ax.set_title(op)
            ax.set_ylabel('|Intensity change|' if j==0 else 'Predicted-class probability decrease');ax.legend()
    fig.tight_layout();save(fig,'fidelity_curves')
    frame=pd.read_csv(OUT3/'validation_predictions_and_explanations.csv')
    fig,axs=plt.subplots(1,2,figsize=(8,3))
    for ax,target in zip(axs,['intensity','classification']):
        counts=frame['main_modality_'+target].value_counts()
        bars=ax.bar([MODALITY_NAMES[m] for m in MODALITIES],[counts.get(m,0) for m in MODALITIES],color=COLORS)
        for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+4,str(int(b.get_height())),ha='center')
        ax.set_ylim(0,len(frame)*1.08);ax.set_ylabel('验证样本数');ax.set_title('解释目标：'+{'intensity':'情感强度','classification':'类别概率'}[target])
    fig.tight_layout();save(fig,'modality_targets')
    from figures_p3_completion import validation_confusion
    validation_confusion()
    cards=[];selection=[]
    for split in ['valid','attachment4']:
        for role,s in choose(split):
            text,row=card(split,role,s);cards.append(text);selection.append(row)
    write_json(OUT3/'cards/selection.json',dict(rule='Per predicted class: highest probability-margin correct and incorrect validation sample; attachment4 highest-margin representative per available class, target disagreement, largest WER. Deduplicate by ID.',samples=selection))
    write_index(OUT3,cards)
    append_completion(OUT3)
    print('Created final-fusion figures and',len(selection),'cards',flush=True)


if __name__=='__main__':main()
