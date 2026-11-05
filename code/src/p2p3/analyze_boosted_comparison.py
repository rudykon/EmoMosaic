"""Paired source-video uncertainty and robustness summaries for compare_boosted."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from compare_boosted import MODEL_NAMES, write_json, write_csv, digest
from data import ROOT, OUT


def cm_metrics(flat):
    cm = np.asarray(flat).reshape(*np.asarray(flat).shape[:-1], 3, 3)
    support = cm.sum(-1)
    predicted = cm.sum(-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = support + predicted
    per_f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float), where=denom>0)
    n = cm.sum((-2,-1))
    return dict(macro_f1=per_f1.mean(-1), accuracy=tp.sum(-1)/n,
                weighted_f1=(per_f1*support).sum(-1)/n)


def grouped_confusion(truth, prediction, groups, n_groups):
    # Rows can include multiple conditions for each clip. Keep all copies of a
    # source video together; do not treat repeated masks as independent samples.
    prediction = np.asarray(prediction).reshape(-1,len(truth))
    codes = groups[None,:]*9 + truth[None,:]*3 + prediction
    return np.bincount(codes.ravel(),minlength=n_groups*9).reshape(n_groups,9).astype(float)


def paired_comparison(a, b, weights, signs=None):
    ma, mb = cm_metrics(a.sum(0)), cm_metrics(b.sum(0))
    ba, bb = cm_metrics(weights@a), cm_metrics(weights@b)
    result={}
    for metric in ('macro_f1','accuracy','weighted_f1'):
        boot=ba[metric]-bb[metric]
        result[metric]=dict(new=float(ma[metric]),baseline=float(mb[metric]),
            delta=float(ma[metric]-mb[metric]),
            delta_ci95=np.quantile(boot,[.025,.975]).tolist(),
            new_ci95=np.quantile(ba[metric],[.025,.975]).tolist(),
            baseline_ci95=np.quantile(bb[metric],[.025,.975]).tolist())
    if signs is not None:
        center=(a.sum(0)+b.sum(0))/2
        shift=signs@(a-b)/2
        perm=cm_metrics(center+shift)['macro_f1']-cm_metrics(center-shift)['macro_f1']
        obs=result['macro_f1']['delta']
        result['macro_f1']['cluster_permutation_p_two_sided']=float(
            (np.count_nonzero(np.abs(perm)>=abs(obs)-1e-14)+1)/(len(perm)+1))
    return result


def holm(pvalues):
    order=np.argsort(pvalues)
    adjusted=np.empty(len(pvalues))
    adjusted[order]=np.minimum(1,np.maximum.accumulate(
        np.asarray(pvalues)[order]*(len(pvalues)-np.arange(len(pvalues)))))
    return adjusted


def load_prediction(output,row):
    return np.load(output/'predictions'/f"{row['split']}_{row['mask_key']}.npz")


def primary_statistics(output,rows,weights,signs,groups,n_groups):
    results=[]
    per_class=[]
    correctness=[]
    checks=[]
    for row in rows[:4]:
        with load_prediction(output,row) as z:
            y=z['truth'];p=z['predictions']
            for j,name in enumerate(MODEL_NAMES):
                cm=confusion_matrix(y,p[j],labels=[0,1,2])
                measured=cm_metrics(cm.ravel())
                assert np.isclose(measured['macro_f1'],f1_score(y,p[j],average='macro',labels=[0,1,2],zero_division=0))
                assert np.isclose(measured['accuracy'],accuracy_score(y,p[j]))
                checks.append(dict(split=row['split'],family=row['family'],model=name,sklearn_matches=True))
                for k,label in enumerate(('Negative','Neutral','Positive')):
                    tp=cm[k,k];support=cm[k].sum();predicted=cm[:,k].sum()
                    per_class.append(dict(split=row['split'],family=row['family'],model=name,
                        label=label,n=int(support),precision=float(tp/max(predicted,1)),
                        recall=float(tp/max(support,1)),f1=float(2*tp/max(support+predicted,1))))
            if row['split']!='test':continue
            for j in (1,2,3,4):
                a=grouped_confusion(y,p[0],groups,n_groups)
                b=grouped_confusion(y,p[j],groups,n_groups)
                comparison=paired_comparison(a,b,weights,signs if j in (1,2) else None)
                results.append(dict(condition=row['family'],baseline=MODEL_NAMES[j],
                                    primary=j in (1,2),metrics=comparison))
                new_correct=p[0]==y;old_correct=p[j]==y
                correctness.append(dict(condition=row['family'],baseline=MODEL_NAMES[j],
                    both_correct=int(np.sum(new_correct&old_correct)),
                    new_only_correct=int(np.sum(new_correct&~old_correct)),
                    baseline_only_correct=int(np.sum(~new_correct&old_correct)),
                    both_wrong=int(np.sum(~new_correct&~old_correct))))
    primary=[r for r in results if r['primary']]
    adj=holm([r['metrics']['macro_f1']['cluster_permutation_p_two_sided'] for r in primary])
    for r,pval in zip(primary,adj):
        r['metrics']['macro_f1']['holm_p_four_primary_comparisons']=float(pval)
    write_json(output/'primary_statistics.json',results)
    write_csv(output/'per_class_metrics.csv',per_class)
    write_csv(output/'correctness_pairs.csv',correctness)
    write_json(output/'independent_metric_checks.json',checks)
    return results


def pooled_statistics(output,rows,weights,groups,n_groups):
    subsets={
        'partial_rate_grid': [r for r in rows if r['split']=='test' and r['family']=='rate' and 0<r['rate']<1],
        'attachment3_replay': [r for r in rows if r['family']=='replay'],
        'mixed_40_five_seeds': [r for r in rows if r['split']=='test' and r['family']=='mixed' and r['rate']==.4],
    }
    results=[]
    for name,subset in subsets.items():
        cms=np.zeros((3,n_groups,9))
        for row in subset:
            with load_prediction(output,row) as z:
                for j in range(3):
                    cms[j]+=grouped_confusion(z['truth'],z['predictions'][j],groups,n_groups)
        for j in (1,2):
            results.append(dict(condition=name,number_of_conditions=len(subset),
                baseline=MODEL_NAMES[j],aggregation='pool confusion matrices across equally weighted conditions, then compute Macro-F1',
                metrics=paired_comparison(cms[0],cms[j],weights)))
    write_json(output/'pooled_statistics.json',results)
    return results


def robustness_tables(output,df):
    test=df[df.split=='test']
    cols=['macro_f1','accuracy','weighted_f1','neutral_recall']
    rate=test[test.family=='rate'].groupby(['model','kind','rate'])[cols].mean().reset_index()
    rate.to_csv(output/'rate_by_type.csv',index=False,encoding='utf-8-sig')
    test[test.family=='position'].groupby(['model','kind','rate','location'])[cols].mean().to_csv(output/'position_summary.csv',encoding='utf-8-sig')
    test[test.family=='duration'].groupby(['model','kind','count'])[cols].mean().to_csv(output/'duration_summary.csv',encoding='utf-8-sig')
    test[test.family=='fragment'].groupby(['model','kind','segments'])[cols].mean().to_csv(output/'fragment_summary.csv',encoding='utf-8-sig')
    test[test.family=='colocated'].groupby(['model','kind','rate','colocated'])[cols].mean().to_csv(output/'colocation_summary.csv',encoding='utf-8-sig')
    test[test.family=='scan'].groupby(['model','kind','location'])[cols].mean().to_csv(output/'scan_summary.csv',encoding='utf-8-sig')
    test[test.family=='mixed'].groupby(['model','rate'])[cols].agg(['mean','std','min','max']).to_csv(output/'mixed_seed_summary.csv',encoding='utf-8-sig')
    parts=test[(test.family=='rate')&(test.rate<1)]
    parts.groupby(['model','kind'])[cols].mean().to_csv(output/'partial_missing_by_type.csv',encoding='utf-8-sig')
    rows=[]
    for (model,kind),sub in rate.groupby(['model','kind']):
        sub=sub.sort_values('rate')
        base=float(test[(test.model==model)&(test.family=='clean')].macro_f1.iloc[0])
        x=np.r_[0.,sub.rate.to_numpy()];y=np.r_[base,sub.macro_f1.to_numpy()]
        rows.append(dict(model=model,kind=kind,clean_macro_f1=base,
            macro_f1_area=float(np.trapezoid(y,x)),
            f1_retention_aurc=float(np.trapezoid(y/base,x)),
            mean_partial_macro_f1=float(sub[sub.rate<1].macro_f1.mean()),
            full_modality_missing_f1=float(y[-1])))
    write_csv(output/'robustness_areas.csv',rows)
    pd.DataFrame(rows).groupby('model').mean(numeric_only=True).to_csv(output/'robustness_average.csv',encoding='utf-8-sig')
    # Compare masks, not duplicated rows across the various diagnostic families.
    unique=parts.drop_duplicates(['model','mask_key']).pivot(index='mask_key',columns='model',values='macro_f1')
    wins=[]
    for baseline in MODEL_NAMES[1:3]:
        delta=unique.svd_lr-unique[baseline]
        wins.append(dict(baseline=baseline,n_masks=len(delta),wins=int((delta>1e-12).sum()),
                         ties=int((delta.abs()<=1e-12).sum()),losses=int((delta<-1e-12).sum()),
                         mean_delta=float(delta.mean())))
    write_json(output/'grid_wins.json',wins)
    return rate


def plot_results(output,rate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'svd_lr':'SVD + logistic regression','rmnet_formal':'RM-Net intensity thresholds',
            'rmnet_aux_probability':'RM-Net auxiliary (mean probabilities)'}
    colors=['#ba4a20','#296d98','#478347']
    df=pd.read_csv(output/'all_metrics.csv')
    fig,axes=plt.subplots(2,4,figsize=(13.5,6),sharex=True,sharey=True)
    for ax,kind in zip(axes.ravel(),('T','A','V','TA','TV','AV','TAV')):
        for (model,label),color in zip(labels.items(),colors):
            sub=rate[(rate.model==model)&(rate.kind==kind)].sort_values('rate')
            base=df[(df.model==model)&(df.split=='test')&(df.family=='clean')].macro_f1.iloc[0]
            ax.plot(np.r_[0,sub.rate],np.r_[base,sub.macro_f1],label=label,color=color,linewidth=1.7)
        ax.set_title(kind);ax.grid(alpha=.2);ax.set_xlabel('Missing fraction');ax.set_ylabel('Macro-F1')
    handles,texts=axes[0,0].get_legend_handles_labels()
    axes[1,3].axis('off');axes[1,3].legend(handles,texts,loc='center',frameon=False,fontsize=8)
    fig.suptitle('Frozen models on 727 held-out test clips; five mask placements per rate')
    fig.tight_layout();fig.savefig(output/'missing_rate_comparison.png',dpi=180)
    fig.savefig(output/'missing_rate_comparison.pdf');plt.close(fig)


def benchmark(output):
    import joblib
    import torch
    import boosted_classifier as boosted
    from compare_boosted import rm_predict
    from data import load_split
    from predict import Predictor
    from train import device_data
    torch.set_num_threads(4)
    model_path=OUT/'boosted_classifier/20260925_225440/model.joblib'
    model=joblib.load(model_path)
    predictor=Predictor(device='cpu')
    data=load_split('test')
    rows=[]
    for n,repeats in ((1,15),(727,3)):
        sub={k:v[:n] for k,v in data.items()}
        tensors=device_data(sub,'cpu')
        for name,fn in (
            ('svd_lr',lambda: boosted.predict(model,sub)),
            ('rmnet_five_seed_ensemble',lambda: rm_predict(predictor,tensors,sub['obs']))):
            fn()
            times=[]
            for _ in range(repeats):
                start=time.perf_counter();fn();times.append(time.perf_counter()-start)
            rows.append(dict(model=name,batch_size=n,repeats=repeats,
                median_ms=float(np.median(times)*1000),min_ms=float(min(times)*1000),
                max_ms=float(max(times)*1000)))
    write_json(output/'cpu_benchmark.json',dict(
        protocol='Sequential runs after evaluation workers exit; CPU float32, four Torch/BLAS threads, one warmup; normalized feature input, no BERT extraction or network/UI time.',
        caveat='SVD predicts polarity only. RM-Net five-seed inference also computes continuous intensity and internal explanation quantities.',
        svd_model_bytes=model_path.stat().st_size,
        rmnet_ensemble_weight_bytes=sum((ROOT/'models/rmnet'/f'weights_seed{s}_fp16.pt').stat().st_size for s in predictor.rules['ensemble_seeds']),
        timings=rows))
    return rows


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def write_report(output,df,primary,pooled):
    names={'svd_lr':'SVD＋逻辑回归','rmnet_formal':'RM-Net 正式阈值',
           'rmnet_aux_probability':'RM-Net 辅助头（平均概率）',
           'rmnet_aux_logits':'RM-Net 辅助头（平均 logits）',
           'svd_lr_before_bias':'SVD＋逻辑回归（无类别偏置）'}
    done=json.loads((output/'inference_complete.json').read_text())
    audit=json.loads((output/'input_audit.json').read_text())
    test=df[(df.split=='test')&(df.audit_scenario_id<4)]
    headline=[]
    for name in MODEL_NAMES:
        clean=test[(test.model==name)&(test.family=='clean')].iloc[0]
        missing=test[(test.model==name)&(test.family=='mixed')].iloc[0]
        headline.append([names[name],f'{clean.accuracy:.4f}',f'{clean.macro_f1:.4f}',
                         f'{missing.accuracy:.4f}',f'{missing.macro_f1:.4f}'])
    stat_rows=[]
    for r in primary:
        if not r['primary']:continue
        m=r['metrics']['macro_f1'];lo,hi=np.array(m['delta_ci95'])*100
        stat_rows.append(['完整输入' if r['condition']=='clean' else '40% 混合缺失',
            names[r['baseline']],f"{m['delta']*100:+.2f}",f'[{lo:+.2f}, {hi:+.2f}]',
            f"{m['cluster_permutation_p_two_sided']:.4f}",f"{m['holm_p_four_primary_comparisons']:.4f}"])
    area=pd.read_csv(output/'robustness_average.csv').set_index('model')
    robustness=[]
    for name in MODEL_NAMES[:4]:
        r=area.loc[name]
        robustness.append([names[name],f'{r.mean_partial_macro_f1:.4f}',
                           f'{r.macro_f1_area:.4f}',f'{r.f1_retention_aurc:.4f}'])
    bytype=pd.read_csv(output/'partial_missing_by_type.csv').pivot(index='kind',columns='model',values='macro_f1')
    type_rows=[]
    for kind in ('T','A','V','TA','TV','AV','TAV'):
        r=bytype.loc[kind]
        type_rows.append([kind,*[f'{r[n]:.4f}' for n in MODEL_NAMES[:3]],
                          f'{100*(r.svd_lr-r.rmnet_formal):+.2f}',
                          f'{100*(r.svd_lr-r.rmnet_aux_probability):+.2f}'])
    pool_rows=[]
    descriptions={'partial_rate_grid':'局部缺失网格（315 条件）',
                  'attachment3_replay':'附件3模式复刻（30 条件）',
                  'mixed_40_five_seeds':'40% 混合缺失（5 个布放种子）'}
    for r in pooled:
        m=r['metrics']['macro_f1'];lo,hi=np.array(m['delta_ci95'])*100
        pool_rows.append([descriptions[r['condition']],names[r['baseline']],
                          f"{m['new']:.4f}",f"{m['baseline']:.4f}",
                          f"{m['delta']*100:+.2f}",f'[{lo:+.2f}, {hi:+.2f}]'])
    perclass=pd.read_csv(output/'per_class_metrics.csv')
    class_rows=[]
    for label in ('Negative','Neutral','Positive'):
        q=perclass[(perclass.split=='test')&(perclass.family=='clean')&(perclass.label==label)].set_index('model')
        class_rows.append([label,int(q.iloc[0].n),*[f'{q.loc[n,"recall"]:.4f}' for n in MODEL_NAMES[:3]]])
    timing=json.loads((output/'cpu_benchmark.json').read_text())
    speed_rows=[[r['model'],r['batch_size'],r['repeats'],f'{r["median_ms"]:.2f}'] for r in timing['timings']]
    full_count=df[['audit_scenario_id','family']].drop_duplicates().groupby('family').size().to_dict()
    wins=json.loads((output/'grid_wins.json').read_text())
    win_text='；'.join(f"相对{names[r['baseline']]}：胜 {r['wins']}、平 {r['ties']}、负 {r['losses']}" for r in wins)
    seed_section=''
    if (output/'seed_sensitivity/complete.json').exists():
        seeds=pd.read_csv(output/'seed_sensitivity/test_metrics.csv')
        seed_rows=[]
        for seed,sub in seeds.groupby('seed'):
            sub=sub.set_index('condition')
            seed_rows.append([int(seed),f'{sub.loc["clean","macro_f1"]:.4f}',f'{sub.loc["missing","macro_f1"]:.4f}'])
        for stat,label in [('mean','五种子均值'),('std','五种子样本标准差')]:
            summary=seeds.groupby('condition').macro_f1.agg(stat)
            seed_rows.append([label,f'{summary["clean"]:.4f}',f'{summary["missing"]:.4f}'])
        seed_section='''## 训练随机种子敏感性（补充检查）

额外固定 7、17、29、43、71 五个训练种子，只改变 SVD 与逻辑回归的随机状态，超参数保持相同。每个种子的类别偏置仅在验证集选择，全部模型保存冻结后才评估测试集。没有按测试分数选择种子；原部署模型仍为 seed 7。该补充方案在主网格运行期间保存，属于探索性稳定性检查。

'''+markdown_table(['训练种子','完整测试 Macro-F1','40% 缺失测试 Macro-F1'],seed_rows)+'''

五次训练共用同一份测试集，不能把它们当作五份独立测试数据。主网格和前述统计区间始终针对原冻结 seed 7 分类器；本表只描述训练随机性。详见 `seed_sensitivity/`。

'''
    all_nonsig=all(r['metrics']['macro_f1']['holm_p_four_primary_comparisons']>=.05 for r in primary if r['primary'])
    conclusion=('主要测试条件下，相对 RM-Net 正式阈值输出只有小幅点估计提升，未达到统计显著；相对 RM-Net 辅助分类头没有点估计优势。不能据此宣称整体识别性能或缺失鲁棒性已提升。'
                if all_nonsig else '主要条件的提升及显著性见下表，应按条件分别解释。')
    text=f'''# SVD＋逻辑回归分类器完整对照测试（2026-09-26）

{conclusion}

## 测试范围与复现核验

- 训练/验证/测试：3,395 / 728 / 727 条；测试集来自 381 个原始视频。三个划分在样本 ID 和原始视频 ID 上均无交集。
- 核验数据缓存、源码、SVD 模型、RM-Net 权重和决策规则 SHA256，与上轮记录一致。APP 模型与冻结 SVD 模型一致。
- 按原训练参数重新训练一次：验证和测试预测分数最大差值均为 0，分类差异均为 0。类别偏置只在验证集选择；测试集仅在训练冻结后评估。
- 归一化统计来自训练集。所有模型接收相同规范化特征、相同有效长度和相同观测掩码；被遮挡位置（含首尾）先置零。
- 使用冻结的五种子 RM-Net 集成，分别评估正式强度阈值、平均 softmax 概率的辅助头、平均 logits 的辅助头；同时评估 SVD 有/无类别偏置。两种辅助集成口径分别列出。
- 覆盖 **{done['scenarios']} 个条件、{done['distinct_masks']} 个不同的有效掩码**。复用项目原有完整网格：七种模态组合，10%–100% 缺失，五个布放种子；另测位置、绝对缺失长度、碎片数、跨模态重合、位置扫描、多种混合缺失和附件3的30种零模式复刻。
- 各类条件数量：`{json.dumps(full_count,ensure_ascii=False,sort_keys=True)}`。相同掩码只推理一次，在相应诊断表保留原协议行。
- GPU 不可用，本轮全部重新使用 CPU float32 推理。正式阈值与平均概率辅助头的四组基础分类预测均逐样本核验原记录；精度差异见 `baseline_reproduction.json`。

## 基础测试集结果

{markdown_table(['模型/判决方式','完整 Accuracy','完整 Macro-F1','40% 缺失 Accuracy','40% 缺失 Macro-F1'],headline)}

40% 混合缺失是在样本间均衡分配 T、A、V、TA、TV、AV、TAV，各受损模态连续遮挡 `ceil(0.4×有效长度)` 个位置。这里列的是原固定种子 20260923。验证集结果保留在 `headline_metrics.csv`，因参与类别偏置选择，不作为独立提升证据。

## 配对差值与统计不确定性

以原始视频为簇，进行 10,000 次配对 bootstrap；同一视频的所有片段、同一片段的各缺失版本一起重采样。四个主要 Macro-F1 对照另做 20,000 次簇级模型预测交换检验，并以 Holm 校正四重比较。差值方向均为 SVD 减基线，单位为百分点。

{markdown_table(['条件','基线','Δ Macro-F1（百分点）','95% 差值区间','双侧置换 p','Holm p'],stat_rows)}

置信区间包含零表示当前样本不足以确认提升，也不能解释为两个模型等效。该区间反映固定模型在视频样本上的不确定性，不涵盖重新训练的随机种子方差。测试网格沿用既有协议，本次补充统计方案在扩展推理前保存；之前的基础分数已知，因此不能称为全新盲测或正式事前注册。

## 全缺失网格的表现

下表的“局部缺失平均 F1”先在每个条件计算 Macro-F1，再对 7 类型 × 9 个比例（10%–90%）× 5 个布放种子等权平均。面积用 0–100% 的 0.1 网格梯形积分，保持率分母是各模型自身的完整输入 F1。100% 是边界压力测试，不等同于任务要求的局部缺失。

{markdown_table(['模型','局部缺失平均 F1','F1 曲线面积','F1 保持率面积'],robustness)}

{markdown_table(['受损模态','SVD F1','RM 正式 F1','RM 辅助 F1','相对正式 Δ（百分点）','相对辅助 Δ（百分点）'],type_rows)}

315 个局部缺失条件的逐条件胜负：{win_text}。这些条件共享同一测试集，胜负次数只能描述覆盖范围，不能当成 315 次独立试验。

![缺失比例对三种预测方式的影响](missing_rate_comparison.png)

下面先合并同一组条件的混淆矩阵，再计算 Macro-F1；它与上面的“各条件 F1 的平均”不是同一个量。簇 bootstrap 保留全部重复遮挡，避免夸大样本量。扩展网格结果属于探索性分析，下面区间未做多重比较校正。

{markdown_table(['条件组','基线','SVD 汇总 F1','基线汇总 F1','Δ（百分点）','95% 差值区间'],pool_rows)}

附件3模式复刻是在 727 条有标签测试样本上施加附件3观察到的30种零模式；包含自然零位置，可能叠加原有无信息位置。这是人工复刻实验，**不是附件3真实准确率**。附件3自身没有标签。

文本先由冻结 BERT 编码，再遮挡特征位置；其余位置可能保留上下文信息。本结论适用于项目规定的预计算特征缺失，不能直接推广到原始文本彻底删除或实时采集效果。

位置、时长、碎片和跨模态重合分别完整保存于 `position_summary.csv`、`duration_summary.csv`、`fragment_summary.csv`、`colocation_summary.csv`、`scan_summary.csv`，没有用这些测试结果改动模型。

## 类别差异与偏置的作用

完整测试集的类别召回率如下：

{markdown_table(['真实类别','样本数','SVD 召回率','RM 正式召回率','RM 辅助召回率'],class_rows)}

无类别偏置版本只作诊断。偏置的目标是验证集 Macro-F1，因此可能提升中性类别召回及 Macro-F1，同时降低总体 Accuracy；不能根据测试 Accuracy 事后把未校正版本选作新正式模型。SVD 的 softmax 概率没有经过独立置信度校准。

{seed_section}
## 运行成本

所有推理工作进程退出后，使用 CPU、4 线程、一次预热，顺序测量规范化特征输入到输出的延迟；不包含 BERT 特征提取、采集、网络或界面耗时。

{markdown_table(['模型','批大小','重复次数','延迟中位数（毫秒）'],speed_rows)}

SVD 模型文件 {timing['svd_model_bytes']/1024**2:.2f} MiB；RM-Net 五种子权重合计 {timing['rmnet_ensemble_weight_bytes']/1024**2:.2f} MiB。RM-Net 同时计算连续强度和解释相关中间量，SVD 只输出极性，能力范围不同。SVD 没有连续情感强度输出，不能据此声称 MAE/Pearson r 改善。

## 复现与原始材料

从项目根目录运行，`--output` 指向新的目录：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \\
  .venv-p2/bin/python src/p2p3/compare_boosted.py \\
  --output outputs/problem2/boosted_classifier/new_full_audit --workers 4 --threads 4

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \\
  .venv-p2/bin/python src/p2p3/analyze_boosted_comparison.py \\
  --output outputs/problem2/boosted_classifier/new_full_audit
```

中断后可对同一目录加 `--resume` 继续推理。`protocol.json`/`scenarios.json` 保存条件，`masks/` 保存掩码，`predictions/` 保存逐样本预测与样本 ID，`all_metrics.csv` 保存全部指标，`primary_statistics.json`/`pooled_statistics.json` 保存区间和检验结果。独立统计校验使用 sklearn 直接重采样与穷举簇交换对照，运行 `test_comparison_statistics.py`。本次结果不修改已部署分类器、RM-Net 参数或原正式预测文件。
'''
    (output/'README.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output
    assert (output/'inference_complete.json').exists()
    rows=json.loads((output/'scenarios.json').read_text())
    with np.load(OUT/'cache/test.npz') as z:
        videos=np.array([str(i).split('$_$')[0] for i in z['ids']])
    names,groups=np.unique(videos,return_inverse=True);n_groups=len(names)
    rng=np.random.default_rng(20260926)
    weights=rng.multinomial(n_groups,np.full(n_groups,1/n_groups),size=10000).astype(float)
    signs=(2*rng.integers(2,size=(20000,n_groups))-1).astype(float)
    primary=primary_statistics(output,rows,weights,signs,groups,n_groups)
    pooled=pooled_statistics(output,rows,weights,groups,n_groups)
    df=pd.read_csv(output/'all_metrics.csv')
    rate=robustness_tables(output,df)
    plot_results(output,rate)
    benchmark(output)
    write_report(output,df,primary,pooled)
    df[df.audit_scenario_id<4].to_csv(output/'headline_metrics.csv',index=False,encoding='utf-8-sig')
    write_json(output/'statistics_complete.json',dict(n_test_clips=len(videos),n_source_videos=n_groups,
        bootstrap_repetitions=10000,cluster_permutations=20000,seed=20260926,
        primary_tests=4,adjustment='Holm',uncertainty='Fixed trained models; uncertainty over sampled source videos, not training seeds.',
        exploratory_grid='Pooled F1 CIs retain all scenario copies per video. Arithmetic mean of scenario F1 is reported separately.',
        source_sha256=digest(__file__)))
    for result in primary:
        if result['primary']:
            print(result['condition'],result['baseline'],result['metrics']['macro_f1'],flush=True)
    print('Statistical analysis and robustness summaries complete.',flush=True)


if __name__=='__main__':
    main()
