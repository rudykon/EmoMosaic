"""Verify all saved adapters against training-time validation predictions."""
import argparse
import numpy as np
from improvement_common import EXP,save_json,score_views,metrics
from improvement_evaluation import CachedEvaluator,base_config
from lora_text_expert import predict_adapter
from train import validation_masks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=4);args=ap.parse_args()
    engine=CachedEvaluator(args.threads);config=base_config();va=engine.dataset('valid')
    masks=[None,validation_masks(va['lengths'],20260923)[1]];results=[];verification=[]
    for seed in [17,29,43,71,101]:
        path=f'outputs/problem2/improvement/lora/seed{seed}/best.pt'
        config['expert']={'kind':'bert_lora','adapters':[path]}
        predictions=[engine.expert(config,'valid',m) for m in masks]
        with np.load(EXP/f'lora/seed{seed}/valid.npz') as old:
            np.testing.assert_array_equal(old['ids'],va['ids'])
            dp=float(np.max(np.abs(old['probabilities']-np.array([r[0] for r in predictions]))))
            dy=float(np.max(np.abs(old['intensity']-np.array([r[1] for r in predictions]))))
            assert dp<2e-6 and dy<2e-6,(seed,dp,dy)
        sub={k:v[:9] for k,v in va.items()}
        model=engine.heads[str(engine.resolve(config,path))]
        direct=predict_adapter(model,sub,masks[1][:9])
        np.testing.assert_allclose(direct['probabilities'],predictions[1][0][:9],atol=2e-6,rtol=1e-5)
        results.append(predictions)
        verification.append({'seed':seed,'saved_validation_max_probability_diff':dp,'saved_validation_max_intensity_diff':dy,
                             'direct_vs_cached_verified':True})
    prob=[np.mean([r[v][0] for r in results],axis=0) for v in [0,1]]
    reg=[np.mean([r[v][1] for r in results],axis=0) for v in [0,1]]
    summary={'primary_score':score_views(va['y'],prob),'clean':metrics(va['y'],prob[0],reg[0]),
             'mixed40':metrics(va['y'],prob[1],reg[1]),'verification':verification}
    save_json(EXP/'lora/ensemble_validation.json',summary)
    np.savez_compressed(EXP/'lora/ensemble_valid.npz',ids=va['ids'],probabilities=prob,intensity=reg)
    print(summary,flush=True)


if __name__=='__main__':main()
