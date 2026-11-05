"""Small train-only diagnostic gates and a bounded two-tool review workflow."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax, expit

from improvement_common import apply_bias, text_observed_fraction


def margin(p):
    ordered = np.sort(p, axis=-1)
    return ordered[:, -1]-ordered[:, -2]


def entropy(p):
    return -(p*np.log(np.clip(p, 1e-12, 1))).sum(-1)/np.log(3)


def quality(data, mask):
    obs = data['obs'] & mask
    length = data['lengths']
    fraction = text_observed_fraction(dict(data, obs=obs))
    values = [fraction, obs[:, 1].sum(1)/length, obs[:, 2].sum(1)/length]
    names = ['text_visible_fraction', 'audio_visible_fraction', 'vision_visible_fraction']
    for j, mod in enumerate(['text', 'audio', 'vision']):
        max_run = []; segments = []
        for row, n in zip(obs[:, j], length):
            absent = ~row[:n]
            if j == 0:
                ids = data['token_ids'][len(max_run), :n]
                absent = absent | (ids == 100)
                absent = absent & (ids != 101) & (ids != 102)
            change = np.diff(np.r_[False, absent, False].astype(int))
            starts = np.flatnonzero(change == 1); ends = np.flatnonzero(change == -1)
            max_run.append(float(np.max(ends-starts, initial=0)/n))
            segments.append(len(starts)/n)
        values.extend([max_run, segments]); names.extend([mod+'_longest_gap', mod+'_gap_count'])
    values.append(length/50); names.append('length_fraction')
    return np.column_stack(values), names


def diagnostic_features(c, review=False):
    p, q = c['rm_p'], c['lora_p']; y, r = c['rm_y'], c['lora_y']
    features = [c['quality'], p, q, (p-q), np.column_stack([
        entropy(p), entropy(q), margin(p), margin(q), (p.argmax(1) != q.argmax(1)),
        y/3, r/3, np.abs(y-r)/6])]
    if review:
        for prefix in ['av', 'text']:
            t = c[prefix+'_p']; s = c[prefix+'_y']
            features.extend([t, t-p, np.column_stack([entropy(t), margin(t), s/3,
                            np.abs(s-y)/6, np.abs(s-r)/6])])
    return np.concatenate(features, axis=1).astype(np.float64)


def baseline(c, rules, shrink=1.):
    f = c['quality'][:, 0]
    a = rules['fusion']['w0']*f**rules['fusion']['k']
    b = rules['regression_w0']*f**rules['regression_k']
    p = apply_bias((1-a[:, None])*c['rm_p']+a[:, None]*c['lora_p'],
                   shrink*np.asarray(rules['fusion']['bias']))
    y = (1-b)*c['rm_y']+b*c['lora_y']
    return p, y


def review_trigger(c, rules):
    p, _ = baseline(c, rules)
    return ((c['quality'][:, 0] > 0) &
            ((c['rm_p'].argmax(1) != c['lora_p'].argmax(1)) | (margin(p) < .20)))


def slice_components(c, indices):
    return {k: v[indices] for k, v in c.items()}


def fit_gate(c, targets, regularization, rules, review=False):
    """Fit standardized linear gates with differentiable probability mixtures.

    Only genuine training OOF predictions enter this function. No row ID,
    target, or previously predicted class is an inference input feature.
    """
    features = diagnostic_features(c, review)
    mean = features.mean(0); scale = np.maximum(features.std(0), .05)
    z = torch.as_tensor(np.column_stack([np.ones(len(features)), (features-mean)/scale]), dtype=torch.float64)
    f = torch.as_tensor(c['quality'][:, 0], dtype=torch.float64)
    labels = torch.as_tensor(np.sign(targets).astype(int)+1)
    target = torch.as_tensor(targets, dtype=torch.float64)
    class_weights = 1/torch.bincount(labels, minlength=3).double().clamp_min(1).sqrt()
    class_weights /= class_weights.mean()
    y_values = torch.as_tensor(np.column_stack([c['rm_y'], c['lora_y'], c['av_y'], c['text_y']])
                               if review else np.column_stack([c['rm_y'], c['lora_y']]), dtype=torch.float64)
    p_values = torch.as_tensor(np.stack([c['rm_p'], c['lora_p'], c['av_p'], c['text_p']], axis=1)
                               if review else np.stack([c['rm_p'], c['lora_p']], axis=1), dtype=torch.float64)
    if review:
        availability = torch.as_tensor(np.column_stack([np.ones(len(f)), f.numpy()**2,
                          np.maximum(c['quality'][:, 1], c['quality'][:, 2])**2, f.numpy()**2]), dtype=torch.float64)
        prior_c = torch.log(torch.tensor([.15, .75, .05, .05], dtype=torch.float64))
        prior_r = torch.log(torch.tensor([.30, .60, .05, .05], dtype=torch.float64))
        shape = (z.shape[1], 4)
    else:
        shape = (z.shape[1],)
        prior_c = float(np.log(.8/.2)); prior_r = float(np.log(.6/.4))
    wc = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float64))
    wr = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float64))
    delta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    base_bias = torch.as_tensor(rules['fusion']['bias'], dtype=torch.float64)
    optimizer = torch.optim.LBFGS([wc, wr, delta], max_iter=180, tolerance_grad=1e-8,
                                 tolerance_change=1e-10, line_search_fn='strong_wolfe')
    calls = 0

    def weights(w, prior):
        if review:
            raw = torch.softmax(z@w+prior, -1)*availability
            return raw/raw.sum(-1, keepdim=True).clamp_min(1e-12)
        a = torch.sigmoid(z@w+prior)*f.square()
        return torch.stack([1-a, a], -1)

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        pc = (p_values*weights(wc, prior_c)[..., None]).sum(1)
        logits = pc.clamp_min(1e-12).log()+base_bias+.5*torch.tanh(delta)
        yr = (y_values*weights(wr, prior_r)).sum(1)
        ce = F.cross_entropy(logits, labels, weight=class_weights)
        reg = F.smooth_l1_loss(yr, target, beta=.5)
        penalty = regularization*(wc.square().mean()+wr.square().mean())+.1*delta.square().mean()
        loss = ce+.3*reg+penalty
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite gate objective')
        loss.backward(); calls += 1
        return loss

    optimizer.step(closure)
    objective = float(closure().detach())
    result = {'kind': 'review_gate' if review else 'adaptive_gate', 'regularization': regularization,
              'mean': mean.tolist(), 'scale': scale.tolist(), 'class_weights': wc.detach().numpy().tolist(),
              'regression_weights': wr.detach().numpy().tolist(),
              'bias': (base_bias+.5*torch.tanh(delta)).detach().numpy().tolist(),
              'objective': objective, 'optimization_calls': calls, 'fit_rows': len(targets),
              'fit_source': 'video-disjoint training OOF', 'feature_count': features.shape[1]}
    return result


def predict_gate(c, gate, rules):
    review = gate['kind'] == 'review_gate'
    x = diagnostic_features(c, review)
    z = np.column_stack([np.ones(len(x)), (x-np.asarray(gate['mean']))/np.asarray(gate['scale'])])
    f = c['quality'][:, 0]
    wc = np.asarray(gate['class_weights']); wr = np.asarray(gate['regression_weights'])
    if review:
        available = np.column_stack([np.ones(len(f)), f**2,
                                     np.maximum(c['quality'][:, 1], c['quality'][:, 2])**2, f**2])
        a = softmax(z@wc+np.log([.15, .75, .05, .05]), -1)*available
        b = softmax(z@wr+np.log([.30, .60, .05, .05]), -1)*available
        a /= a.sum(1, keepdims=True); b /= b.sum(1, keepdims=True)
        p = (np.stack([c['rm_p'], c['lora_p'], c['av_p'], c['text_p']], 1)*a[..., None]).sum(1)
        y = (np.column_stack([c['rm_y'], c['lora_y'], c['av_y'], c['text_y']])*b).sum(1)
    else:
        a = expit(z@wc+np.log(.8/.2))*f**2
        b = expit(z@wr+np.log(.6/.4))*f**2
        p = (1-a[:, None])*c['rm_p']+a[:, None]*c['lora_p']
        y = (1-b)*c['rm_y']+b*c['lora_y']
    p = apply_bias(p, gate['bias'])
    absent = f == 0
    if absent.any():
        p[absent] = apply_bias(c['rm_p'][absent], rules['fusion']['bias'])
        y[absent] = c['rm_y'][absent]
    return p, y


def predict_candidate(c, candidate, rules):
    kind = candidate['kind']
    triggered = np.zeros(len(c['rm_y']), dtype=bool)
    if kind == 'baseline': p, y = baseline(c, rules)
    elif kind == 'bias_shrink': p, y = baseline(c, rules, candidate['shrink'])
    elif kind == 'adaptive': p, y = predict_gate(c, candidate['gate'], rules)
    elif kind == 'review':
        p, y = predict_gate(c, candidate['adaptive'], rules)
        triggered = review_trigger(c, rules)
        if triggered.any():
            p[triggered], y[triggered] = predict_gate(slice_components(c, triggered), candidate['gate'], rules)
    else: raise ValueError(kind)
    return p, np.clip(y, -3, 3), triggered
