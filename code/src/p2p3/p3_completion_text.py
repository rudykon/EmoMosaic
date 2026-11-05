"""Evidence-backed additions shared by the paper and the P3 report generator."""
import json
import textwrap
import numpy as np
from p3_common import OUT3

LABEL = {'Negative': '负', 'Neutral': '中', 'Positive': '正'}
MOD = {'text': 'T', 'audio': 'A', 'vision': 'V'}


def error_analysis(heading, para, table, figure, load, figure_ref, begin_card, end_card):
    heading('验证集错误分析', 3)
    figure('validation_confusion', '验证集混淆矩阵与分类指标。最终融合模型在同一批728条验证样本上的结果；(a)行、列分别为真实类别与预测类别，格内为样本数；(b)为各类别精确率、召回率和F1，柱顶数值以百分数表示，n为真实样本数。')
    para('文本与视觉冲突定义为二者对原预测类概率的Shapley值异号。冲突组357条的错误率为38.94%，高于其余371条的32.35%，相差6.59个百分点；这表明该分组适合优先人工回看，但不能据此认定冲突导致误判。概率间隔最低两个四分位的错误率分别为48.90%和50.55%，最高四分位为14.84%，整体呈现较高间隔对应较少分类错误的趋势，但并非逐层单调。语音主导组只有7条，其57.14%的错误率不足以支持模态优劣的总体判断。')
    ids = ['f_ZJ7L14oYQ$_$21', 'EQxkXiWAm08$_$8', 'x2lOwQaAn4g$_$4']
    samples = [load('samples/valid/' + sid + '.json') for sid in ids]
    rows = []
    for i, s in enumerate(samples):
        p = s['prediction']; truth_label = ['负', '中', '正'][int(np.sign(s['truth']) + 1)]
        rows.append([f'E{i+1}', truth_label, f"{s['truth']:.4f}", LABEL[p['pred_polarity']],
                     f"{p['pred_intensity']:.4f}", f"{p['probability_margin']:.4f}"])
    table(['案例', '真实极性', '真实强度', '预测极性', '预测强度', '概率间隔'], rows,
          caption='三类预测中的代表性误判及真实标签', label='tab:p3-errors')
    para('以下解释卡按同一官方词元轴分层展示文字、文本贡献、语音特征与贡献、视觉特征与贡献。验证集未提供原始素材，横轴不表示秒数，特征热图也不冒充原始波形或视频帧。橙蓝贡献轨迹分别表示支持和抵消原预测类，紫绿热图仅表示标准化输入特征的正负；各贡献轨迹的倍率在右侧标明。卡片用于对照模型依赖与真实标签，不将局部证据直接认定为误判原因。')
    def error_card(i):
        s=samples[i];p=s['prediction'];name=f"card_valid_{p['id']}".replace('$','')
        truth_label=['负向','中性','正向'][int(np.sign(s['truth'])+1)]
        predicted={'Negative':'负向','Neutral':'中性','Positive':'正向'}[p['pred_polarity']]
        figure_ref(name,f'E{i+1}的序列特征与局部证据见')
        figure(name,f"误判E{i+1}解释卡（{p['id']}）：真实{truth_label}，预测{predicted}；真实强度{s['truth']:.4f}，预测强度{p['pred_intensity']:.4f}。")
        end_card()
    begin_card()
    para('E1对应验证样本f_ZJ7L14oYQ$_$21。原句见对应本地样本记录。真实强度为2.0，评价对象被称为更方便，而负面措辞主要指向被比较的boots。模型却输出负类及−1.6812的强度，提示评价对象与比较范围可能被混淆。局部正向支持负类的前三个词元来自aren\'t和两处just，其概率下降仅约0.00331、0.00262、0.00239；这支持对上下文与比较结构作进一步检查，不能直接把stupid一词认定为已证实的误判原因。')
    error_card(0)
    begin_card()
    para('E2对应EQxkXiWAm08$_$8，原句见对应本地样本记录。真实强度0.3333属于弱正，模型强度0.1004也为正，但独立分类路径判为中性。句子陈述事务性要求，标注弱正与模型中性之间的边界是主要误差表现；不能用回归值为正来代替最终分类，也不能据一句案例改动中性定义。')
    error_card(1)
    begin_card()
    para('E3对应x2lOwQaAn4g$_$4，原句见对应本地样本记录。真实标签为中性，模型预测正类且强度1.6745，说明便利性描述被整体预测为明显正向。其排名第一的guarantee的局部概率下降为−0.002278，即遮挡后正类概率反而略升；该词在所定义的局部干预下抵消原类别，不能写成推动误判的证据。单点排序必须结合符号、整句上下文及多位置干预解释，整体正向预测也不要求每个词都具有正贡献。')
    error_card(2)
    para('以上分别定位了比较对象、弱正与中性边界、整体语义与单点证据方向三个待复核环节。它们是由真值、原句和模型干预支持的错误诊断假设，尚未经过人工语义标注或专门对照验证。')


def detailed_replay(heading, para, table, figure, load):
    heading('完整局部分布与可信候选定位', 3)
    para('选择19号作为双目标对照：最终预测为负向，强度为0.0335；分类主要参考文本，强度主要参考视觉。其分类Shapley值按T/A/V分别为0.25394、−0.05226、−0.02965，文本提高负类概率，音频与视觉则抵消负类；强度Shapley分别为−0.25695、0.01894、0.26464，文本的负向效应与视觉的正向效应相互抵消。各值均相对于模型全空输入基线，说明模态作用依赖所解释的目标，不能仅凭主模态名称判定预测方向。')
    para('下列两幅图覆盖该样本全部39个有效内容位置，按原词元顺序展示分类和强度的单点遮挡效应。颜色在每个模态内除以其最大绝对效应，行标签保留原始尺度；该归一只用于观察模态内部的分布，不能按不同模态颜色深浅比较作用大小。正值表示该位置提高对应输出，负值表示降低。主要参考模态标星，特殊词元不参与排序。')
    figure('local_distribution_attachment4_19_classification',
           '19号原预测类概率的完整局部重要性分布。文本是分类主模态；颜色按模态内最大绝对效应缩放，行标签max给出原始尺度。', height='.53')
    figure('local_distribution_attachment4_19_intensity',
           '19号融合强度的完整局部重要性分布。视觉是强度主模态；位置与前图一致，正负值分别对应强度增加与降低。', height='.53')
    s = load('samples/attachment4/19.json')
    p = s['prediction']; target = s['fixed_target_class']; baseline = [p['p_neg'], p['p_neu'], p['p_pos']][target]
    rows = []
    for m in ['text', 'audio', 'vision']:
        r = s['top_evidence'][m][0]
        location = f"字符[{r['char_start']},{r['char_end']})" if m == 'text' else (
            f"{r['time_start']:.3f}—{r['time_end']:.3f}秒" if m == 'audio' else
            f"帧{r['keyframe']['frame_index']}\nPTS {r['keyframe']['pts_relative']:.3f}秒")
        rows.append([MOD[m], f"{r['token']} [{r['position']}]", location,
                     f"{r['class_probability_delta']:+.6f}", f"{baseline-r['class_probability_delta']:.6f}"])
    para(f'19号完整输入的负类概率为{baseline:.6f}。下表对各模态分类排序第一的位置给出定位与遮挡后的同一类别概率。文本位置17的子词hate属于原词haters，并非独立原词；字符区间采用从0开始的左闭右开索引，音视频采用原素材秒数与从0开始的帧号。候选时间均通过既定质量阈值。')
    table(['模态', '词元[位置]', '素材定位', '概率下降', '遮挡后概率'], rows,
          caption='19号关键证据的原素材定位与干预结果', label='tab:p3-replay')
    figure('source_replay_attachment4_19',
           '19号三模态证据回溯：原句及文本遮挡效应、音频RMS与候选区间、视觉原帧与PTS。RMS仅辅助回看，归因来自官方特征；全部时间仍为独立CTC候选估计。', height='.66')
    para('15、16号用于说明低可信定位边界，不将候选时间写成精确真值。13号的官方视觉观测没有可排序内容位置，因此视觉证据为空；01号的独立人脸检出率虽然为零，但不能据此否定其全部官方视觉特征。模型观测可用性与独立素材核验质量分别记录。')


def full_results(heading, para, table, frame):
    heading('附件4全量预测与解释汇总')
    para('附件4全部20条样本的极性、强度及模态作用如下。负向8条、中性5条、正向7条；分类主模态均为文本，强度主模态16条为文本、4条为视觉。T、A、V依次表示文本、语音、视觉；分类作用占比为原预测类Shapley绝对值归一化，不表示正向支持比例，也不是概率。附件4没有真实标签，以下仅为最终预测与解释汇总。')
    rows = []
    for r in frame.to_dict('records'):
        rows.append([r['id'], LABEL[r['pred_polarity']], f"{r['pred_intensity']:.4f}",
                     MOD[r['main_modality_classification']] + '/' + MOD[r['main_modality_intensity']],
                     *[f"{r['share_class_'+m]:.3f}" for m in 'TAV']])
    table(['编号', '极性', '强度', '主模态\n类别/强度', 'T占比', 'A占比', 'V占比'], rows,
          caption='附件4全部20条最终预测与模态作用', label='tab:p3-all-predictions')
    para('下表压缩展示每条样本各模态分类排序第一的证据；位置均为官方对齐序列中的词元位置，时间单位为秒，帧号从0开始。文本列给出所属原词，完整子词与字符偏移保存在逐样本结果中。“低”表示未通过时间质量阈值，数值只作候选回看；“无”表示没有可用观测证据。前三位置及强度目标的完整证据保留于补充结果中。')
    rows = []
    for path in sorted((OUT3 / 'samples/attachment4').glob('*.json')):
        s = json.loads(path.read_text()); t = s['top_evidence']['text'][0]; a = s['top_evidence']['audio'][0]
        vv = s['top_evidence']['vision']; v = vv[0] if vv else None
        word = '\n'.join(textwrap.wrap(t['word'], width=15, break_long_words=True, break_on_hyphens=False))
        vision = f"{v['keyframe']['frame_index']} / {v['keyframe']['pts_relative']:.3f}" if v else '无'
        quality = '/'.join('可' if r and r['timing_usable'] else ('低' if r else '无') for r in [a, v])
        rows.append([s['prediction']['id'], word + f" [{t['position']}]",
                     f"{a['time_start']:.3f}—{a['time_end']:.3f}", vision, quality])
    table(['编号', '文本原词[位置]', '语音区间', '视觉帧/PTS', '时间质量\nA/V'], rows,
          caption='附件4全部20条关键证据定位', label='tab:p3-all-evidence')
    para('上述两表分别汇总最终预测、模态作用与关键证据定位。补充结果按样本编号保留三类概率、两个目标的Shapley值、全部位置的遮挡效应、八子集输出及类别翻转过程；原词、候选时间、原始帧号与质量标志共同构成证据的溯源信息。')
