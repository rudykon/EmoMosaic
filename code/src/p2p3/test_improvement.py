"""Scientific invariants: missing-token leakage, exact fallback, selection metric."""
import unittest

import numpy as np
from sklearn.metrics import f1_score

from improvement_common import pool_tokens, fusion, score_views, select_bias, apply_bias, text_observed_fraction
from probe_text_experts import llm_observed
from extract_text_experts import decode_with_offsets
from strict_text_intervention import delete_spans


class ImprovementTests(unittest.TestCase):
    def test_pool_excludes_occluded_and_padding_extrema(self):
        x=np.array([[[1.,2.],[999.,-999.],[3.,4.]]])
        observed=np.array([[True,False,True]])
        expected=np.array([[2.,3.,3.,4.,1.,2.]])
        np.testing.assert_allclose(pool_tokens(x,observed),expected)
        np.testing.assert_array_equal(pool_tokens(x,np.zeros_like(observed)),np.zeros((1,6)))

    def test_llm_alignment_rejects_any_overlap(self):
        offsets=np.array([[0,0],[0,2],[2,5],[5,8],[8,10]])
        bert=np.array([[0,0],[0,4],[4,8],[0,0]])
        np.testing.assert_array_equal(llm_observed(offsets,bert,[True,False,True,True]),[False,False,False,True,True])
        self.assertFalse(llm_observed(offsets,bert,[False]*4).any())

    def test_text_absence_disables_expert_even_k_zero(self):
        expert=np.array([[.99,.005,.005],[.1,.2,.7]])
        core=np.array([[.1,.7,.2],[.3,.4,.3]])
        for k in [0.,1.,2.,4.]:
            p,w=fusion(expert,core,[0.,.5],.8,k)
            np.testing.assert_allclose(p[0],core[0])
            self.assertEqual(w[0],0.)
            np.testing.assert_allclose(p.sum(1),1.)

    def test_unknown_content_is_not_reliable_despite_cls_sep(self):
        data={'token_ids':np.array([[101,100,100,102,0],[101,1045,100,102,0]]),
              'obs':np.ones((2,3,5),bool)}
        np.testing.assert_array_equal(text_observed_fraction(data),[0.,.5])

    def test_strict_deletion_does_not_join_separate_words(self):
        self.assertEqual(delete_spans('not very good',[[0,3],[4,8],[9,13]],[True,False,True]),'not      good')
        self.assertEqual(delete_spans('not good',[[0,3],[4,8]],[False,False]),'')

    def test_primary_is_mean_of_view_f1(self):
        y=np.array([-1.,-1.,0.,1.,1.])
        predictions=[np.array([0,0,1,2,1]),np.array([1,1,1,1,2])]
        p=[np.eye(3)[r]*.98+.02/3 for r in predictions]
        expected=np.mean([f1_score(np.sign(y)+1,r,average='macro',labels=[0,1,2]) for r in predictions])
        self.assertAlmostEqual(score_views(y,p),expected)
        selected=select_bias(y,p)
        actual=score_views(y,[apply_bias(v,selected['bias']) for v in p])
        self.assertAlmostEqual(actual,selected['score'])
        self.assertGreaterEqual(actual,expected)

    def test_supplied_unknown_and_subword_positions_survive_decoding(self):
        from transformers import AutoTokenizer
        from data import ROOT, OUT
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/bert',local_files_only=True)
        with np.load(OUT/'cache/attachment3.npz') as z:
            ids=z['token_ids']
        unknowns=0
        for row in ids:
            text,offsets=decode_with_offsets(row,tokenizer)
            for token,(start,end) in zip(row,offsets):
                if token==tokenizer.unk_token_id:
                    self.assertEqual(text[start:end],'[UNK]')
                    unknowns+=1
                if token in [tokenizer.pad_token_id,tokenizer.cls_token_id,tokenizer.sep_token_id]:
                    self.assertEqual(start,end)
        self.assertEqual(unknowns,131)

    def test_layer_mixture_backprop_and_mask_invariance(self):
        import torch
        import yaml
        from data import ROOT
        from model import RMNet
        torch.set_num_threads(1)
        torch.manual_seed(20260926)
        cfg=yaml.safe_load((ROOT/'src/p2p3/config.yaml').read_text())
        cfg.update(text_layers=13,hidden=16,heads=2,layers=1)
        model=RMNet(cfg).eval()
        x=[torch.randn(2,13,50,768),torch.randn(2,50,74),torch.randn(2,50,35)]
        observed=torch.ones(2,3,50,dtype=torch.bool)
        lengths=torch.full((2,),50)
        mask=observed.clone();mask[:,0,5:10]=False
        first=model(x,observed,lengths,mask)
        altered=[v.clone() for v in x];altered[0][:,:,5:10]=10000
        second=model(altered,observed,lengths,mask)
        torch.testing.assert_close(first['y'],second['y'])
        torch.testing.assert_close(first['logits'],second['logits'])
        first['logits'].square().sum().backward()
        grad=model.text_layer_logits.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0)

    def test_lora_updates_only_rank_matrices_and_matches_merged_equation(self):
        import torch
        from lora_text_expert import LoRALinear
        torch.manual_seed(41)
        base=torch.nn.Linear(7,5)
        old_weight=base.weight.detach().clone();old_bias=base.bias.detach().clone()
        layer=LoRALinear(base,rank=2,alpha=4,dropout=0.)
        x=torch.randn(3,7)
        torch.testing.assert_close(layer(x),base(x))
        opt=torch.optim.SGD([layer.a,layer.b],lr=.1)
        layer(x).square().sum().backward();opt.step()
        torch.testing.assert_close(base.weight,old_weight);torch.testing.assert_close(base.bias,old_bias)
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.b.abs().sum().item(),0.)
        merged=torch.nn.functional.linear(x,old_weight+layer.scale*(layer.b@layer.a),old_bias)
        torch.testing.assert_close(layer(x),merged)

    def test_evaluation_scenarios_cover_replays_and_reject_misalignment(self):
        import json
        from data import load_split,OUT
        from evaluate_improvement import scenarios
        d=load_split('valid');d={k:v[:3] for k,v in d.items()}
        special=load_split('attachment3')
        replays=json.loads((OUT/'data_audit.json').read_text())['special']['attachment3']
        rows=list(scenarios(d,replays,special))
        self.assertEqual(len(rows),85)
        self.assertEqual(len(set(r[0] for r in rows)),85)
        by_name={name:mask for name,_,mask in rows}
        for index in range(30):
            zero=by_name[f'replay_zero_{index:02d}'];unk=by_name[f'replay_zero_unk_{index:02d}']
            self.assertFalse(np.any(unk&~zero))
            np.testing.assert_array_equal(unk[:,1:],zero[:,1:])
        with self.assertRaisesRegex(ValueError,'orders differ'):
            list(scenarios(d,replays[::-1],special))

    def test_cluster_statistics_match_sklearn_and_identical_models(self):
        from analyze_boosted_comparison import grouped_confusion,paired_comparison
        truth=np.array([0,0,1,1,2,2,2]);predicted=np.array([0,1,1,2,2,0,2])
        groups=np.array([0,0,1,1,2,2,2]);cm=grouped_confusion(truth,predicted,groups,3)
        rng=np.random.default_rng(71);weights=rng.multinomial(3,[1/3]*3,size=500)
        signs=rng.choice([-1,1],size=(500,3));r=paired_comparison(cm,cm,weights,signs)
        self.assertAlmostEqual(r['macro_f1']['new'],f1_score(truth,predicted,average='macro',labels=[0,1,2]))
        np.testing.assert_array_equal(r['macro_f1']['delta_ci95'],[0,0])
        self.assertEqual(r['macro_f1']['cluster_permutation_p_two_sided'],1.)
        from evaluate_improvement import paired_mean_views
        alternate=np.array([1,1,1,1,2,1,2]);second=grouped_confusion(truth,alternate,groups,3)
        average=paired_mean_views([cm,second],[cm,second],weights,signs)
        expected=np.mean([f1_score(truth,v,average='macro',labels=[0,1,2]) for v in [predicted,alternate]])
        self.assertAlmostEqual(average['macro_f1']['new'],expected)
        np.testing.assert_array_equal(average['macro_f1']['delta_ci95'],[0,0])


if __name__=='__main__':
    unittest.main()
