"""Audit supplied inputs and cache ONE frozen BERT feature version for all splits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from project_paths import DATA_ROOT
from data import ROOT, OUT, MODALITIES, load_pickle, unwrap, raw_arrays, write_json, zero_runs, normalize


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output-root', type=Path, default=OUT,
                    help='Directory for cache/ and data_audit.json; use a fresh directory to rebuild')
    ap.add_argument('--refit-normalization', action='store_true',
                    help='Fit train-only statistics for a new training run, saved below output-root')
    args = ap.parse_args()
    out = args.output_root.resolve()
    expected = [out/'cache'/f'{s}.npz' for s in ['train','valid','test','attachment3','attachment4']]
    expected += [out/'cache'/f'{s}_text_cache.npz' for s in ['attachment3','attachment4']]
    if all(path.exists() for path in expected) and not args.refit_normalization:
        print(f'All seven prepared cache files already exist at {out / "cache"}; no files changed.')
        return
    if any(path.exists() for path in expected):
        ap.error('Partial/existing caches will not be overwritten. Select a fresh --output-root.')
    if args.refit_normalization and out == OUT:
        ap.error('New normalization requires a fresh --output-root to preserve deployed assets.')
    (out/'cache').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    from transformers import BertModel
    cfg = yaml.safe_load((ROOT / 'src/p2p3/config.yaml').read_text())
    model = BertModel.from_pretrained(ROOT/'models/pretrained/bert',
                                     local_files_only=True, use_safetensors=True).eval().to(args.device)
    for p in model.parameters(): p.requires_grad_(False)
    src = DATA_ROOT/'source_data/attachment2_features/aligned_50.pkl'
    dataset = load_pickle(src)
    audit = {'source_sha256': sha256(src), 'text_policy': cfg['text_feature_policy'],
             'bert_revision': cfg['text_revision'], 'splits': {}, 'special': {}}
    prepared = {}
    @torch.inference_mode()
    def encode(d):
        t = np.asarray(d['text_bert']); t = t[None] if t.ndim == 2 else t
        out = []
        for s in range(0, len(t), 64):
            b = torch.as_tensor(t[s:s+64], dtype=torch.long, device=args.device)
            observed = (b[:, 1] != 0) & (b[:, 0] != 0)
            safe = observed.clone(); safe[~safe.any(1), 0] = True
            with torch.autocast('cuda', dtype=torch.float16, enabled='cuda' in args.device):
                x = model(input_ids=b[:, 0], attention_mask=safe, token_type_ids=b[:, 2]).last_hidden_state
            out.append((x.float() * observed[..., None]).cpu().numpy())
        return np.concatenate(out)
    for split, d in dataset.items():
        print('encoding', split, len(d['text_bert']), flush=True)
        text = encode(d)
        arrays, obs, lengths = raw_arrays(d, text)
        y = np.asarray(d['regression_labels'], np.float32).reshape(-1)
        valid = np.arange(50)[None] < lengths[:, None]
        original = np.asarray(d['text'])
        info = {'n': len(y), 'length_quantiles': np.quantile(lengths, [0,.25,.5,.75,1]).tolist(),
                'classes_negative_neutral_positive': [(np.sign(y)==c).sum().item() for c in [-1,0,1]],
                'original_text_nonzero_padding_fraction': float(np.any(original[~valid]!=0,-1).mean()),
                'original_text_vs_frozen_bert_mae_observed': float(np.abs(original[valid]-text[valid]).mean()),
                'cls_ids': np.unique(np.asarray(d['text_bert'])[:,0,0]).tolist(), 'modalities': {}}
        for j,m in enumerate(MODALITIES):
            x = arrays[j]; rows=x[obs[:,j]]
            info['modalities'][m]={'natural_zero_fraction': float((~obs[:,j])[valid].mean()),
                'nonfinite_raw': int((~np.isfinite(np.asarray(d[m]))).sum()),
                'absolute_quantiles': np.quantile(np.abs(rows), [.5,.9,.99,.999,1]).tolist(),
                'boundary_zero_fraction': float((~obs[:,j,0]).mean())}
        audit['splits'][split]=info
        prepared[split]=(arrays, obs, lengths, y, d)
    train_arrays, train_obs, *_ = prepared['train']
    stats = {}
    for j,m in enumerate(MODALITIES):
        rows=train_arrays[j][train_obs[:,j]].astype(np.float64)
        stats[m]={'mean': rows.mean(0).tolist(), 'std': np.maximum(rows.std(0), 1e-5).tolist(), 'n_observed':len(rows)}
    if args.refit_normalization:
        write_json(out/'norm_stats.json', stats)
        audit['normalization_policy'] = 'new_train_only_statistics'
    else:
        stats_path = ROOT/'models/rmnet/norm_stats.json'
        if not stats_path.is_file():
            raise FileNotFoundError('Frozen norm_stats.json missing; use --refit-normalization for a new training run')
        stats = json.loads(stats_path.read_text())
        audit['normalization_policy'] = 'reuse_frozen_training_statistics'
        audit['normalization_sha256'] = sha256(stats_path)
    for split,(arrays,obs,lengths,y,d) in prepared.items():
        xx=normalize(arrays,obs,stats)
        np.savez(out/'cache'/f'{split}.npz', **dict(zip(MODALITIES,xx)), obs=obs, lengths=lengths,
                 y=y, ids=np.asarray(d['id'],dtype=str), raw_text=np.asarray(d['raw_text'],dtype=str),
                 token_ids=np.asarray(d['text_bert'])[:,0].astype(np.int64))
    for name, pattern in [('attachment3','attachment3*/aligned/*.pkl'),('attachment4','attachment4*/*/aligned/*.pkl')]:
        records=[]; all_x=[[],[],[]]; all_obs=[]; all_l=[]; all_t=[]
        paths=sorted((DATA_ROOT / 'source_data').glob(pattern))
        for path in paths:
            d=unwrap(load_pickle(path)); text=encode(d)
            arrays,obs,lengths=raw_arrays(d,text); xx=normalize(arrays,obs,stats)
            for j in range(3): all_x[j].append(xx[j])
            all_obs.append(obs); all_l.append(lengths)
            tok=np.asarray(d['text_bert']);tok=tok[None] if tok.ndim==2 else tok
            all_t.append(tok[:,0].astype(np.int64))
            record={'file':path.name,'sample_id':str(d.get('id',path.stem)), 'valid_len':int(lengths[0]),
                    'input_keys':list(d), 'sha256':sha256(path)}
            for j,m in enumerate(MODALITIES):
                record[f'miss_{m}_intervals']=zero_runs(~obs[0,j,:lengths[0]])
                record[f'miss_ratio_{m}']=float(1-obs[0,j].sum()/lengths[0])
            records.append(record)
        audit['special'][name]=records
        np.savez(out/'cache'/f'{name}.npz', **{m:np.concatenate(all_x[j]) for j,m in enumerate(MODALITIES)},
                 obs=np.concatenate(all_obs),lengths=np.concatenate(all_l), token_ids=np.concatenate(all_t),
                 ids=np.asarray([r['sample_id'] for r in records]),files=np.asarray([r['file'] for r in records]))
        # Small frozen text cache permits offline reproduction of supplied special sets.
        np.savez_compressed(out/'cache'/f'{name}_text_cache.npz',
             text=np.concatenate(all_x[0]).astype(np.float16),
             sha256=np.asarray([r['sha256'] for r in records]),
             files=np.asarray([r['file'] for r in records]))
    write_json(out/'data_audit.json',audit)
    print('Prepared all splits and both special sets.', flush=True)


if __name__=='__main__':main()
