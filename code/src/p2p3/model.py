"""RM-Net: observed-key cross attention, reliability gates, bounded additive output."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(logits, mask, dim=-1):
    p=torch.softmax(logits.float().masked_fill(~mask,-1e4),dim=dim)*mask
    return p / p.sum(dim,keepdim=True).clamp_min(1e-8)


class Attention(nn.Module):
    def __init__(self,d,heads,drop):
        super().__init__();self.heads=heads;self.drop=drop
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
    def forward(self,q,kv,mask,bias=None):
        B,L,D=q.shape;K=kv.shape[1];H=self.heads
        qq=self.q(q).reshape(B,L,H,D//H).transpose(1,2)
        kk=self.k(kv).reshape(B,K,H,D//H).transpose(1,2)
        vv=self.v(kv).reshape(B,K,H,D//H).transpose(1,2)
        b=torch.zeros((B,1,L,K),device=q.device,dtype=q.dtype)
        b=b.masked_fill(~mask[:,None,None,:],float('-inf'))
        if bias is not None:b=b+bias[None,None].to(b.dtype)
        y=F.scaled_dot_product_attention(qq,kk,vv,attn_mask=b,dropout_p=self.drop if self.training else 0)
        return self.out(y.transpose(1,2).reshape(B,L,D))


class Block(nn.Module):
    def __init__(self,d,h,drop,mult=2):
        super().__init__();self.attn=Attention(d,h,drop)
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.drop=nn.Dropout(drop)
        self.ff=nn.Sequential(nn.Linear(d,d*mult),nn.GELU(),nn.Dropout(drop),nn.Linear(d*mult,d))
    def forward(self,x,mask,kv=None,bias=None):
        q=self.n1(x)
        x=x+self.drop(self.attn(q,q if kv is None else kv,mask,bias))
        return x+self.drop(self.ff(self.n2(x)))


class RMNet(nn.Module):
    def __init__(self,cfg,variant='A0'):
        super().__init__();self.cfg=cfg;self.variant=variant
        d=cfg['hidden'];dims=cfg['dimensions'];drop=cfg['dropout']
        self.input_dropout=nn.Dropout(cfg['input_dropout'])
        self.proj=nn.ModuleList([nn.Linear(v,d) for v in dims])
        self.miss=nn.Parameter(torch.zeros(3,d));self.absent=nn.Parameter(torch.zeros(3,d))
        self.position=nn.Parameter(torch.randn(50,d)*.02)
        self.encoders=nn.ModuleList([nn.ModuleList([Block(d,cfg['heads'],drop,cfg['ff_mult'])
                                  for _ in range(cfg['layers'])]) for _ in dims])
        self.cross=nn.ModuleList([Block(d,cfg['heads'],drop,cfg['ff_mult']) for _ in dims])
        self.align_strength=nn.Parameter(torch.ones(3))
        self.decoder=nn.ModuleList([nn.Linear(d,v) for v in dims])
        self.pool=nn.ModuleList([nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1)) for _ in dims])
        self.head=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d//2),nn.GELU(),
                                            nn.Dropout(drop),nn.Linear(d//2,1)) for _ in dims])
        self.gate=nn.ModuleList([nn.Sequential(nn.Linear(d+3,d//2),nn.GELU(),nn.Linear(d//2,1)) for _ in dims])
        self.interaction=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,1))
        self.classifier=nn.Linear(d,3)
        self.empty=nn.Parameter(torch.zeros(()))
        if cfg.get('text_layers'):
            self.text_layer_logits=nn.Parameter(torch.zeros(cfg['text_layers']))
        if variant in ('A4','B_ConcatClean'):self.concat=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
        if variant=='B_LMF':
            self.factors=nn.ParameterList([nn.Parameter(torch.randn(d+1,cfg.get('lmf_rank',8),cfg.get('lmf_latent',16))*.03) for _ in dims])
            self.lmf_head=nn.Linear(cfg.get('lmf_latent',16),1)
        distance=(torch.arange(50)[:,None]-torch.arange(50)[None]).abs().float()
        self.register_buffer('distance',torch.cat([distance,distance,torch.zeros(50,1)],1),persistent=False)

    def forward(self,x,obs,lengths,mask=None):
        if x[0].ndim==4:
            if not hasattr(self,'text_layer_logits') or x[0].shape[1]!=len(self.text_layer_logits):
                raise ValueError('Layer feature shape does not match configured scalar mixture')
            mixed=torch.einsum('bltd,l->btd',x[0],self.text_layer_logits.softmax(0))
            x=[mixed,*x[1:]]
        obs=obs.bool()
        if mask is not None:obs=obs & mask.bool()
        valid=torch.arange(50,device=lengths.device)[None,:]<lengths[:,None]
        obs=obs & valid[:,None,:]
        if self.variant in ('B_T','B_A','B_V'):
            select='TAV'.index(self.variant[-1]);obs=obs.clone()
            for m in range(3):
                if m!=select:obs[:,m]=False
        hs=[]
        for m in range(3):
            xx=x[m]*obs[:,m,:,None]
            h=self.proj[m](self.input_dropout(xx))*obs[:,m,:,None]
            if self.variant!='A6':h=h+(~obs[:,m,:,None])*self.miss[m]
            h=h+self.position
            for block in self.encoders[m]:h=block(h,valid)
            hs.append(h)
        interacted=[]
        for m in range(3):
            if self.variant=='A8' or self.variant.startswith('B_'):h=hs[m]
            else:
                others=[j for j in range(3) if j!=m]
                kv=torch.cat([hs[j] for j in others]+[torch.zeros_like(hs[m][:,:1])],1)
                keys=torch.cat([obs[:,j] for j in others],1)
                keys=torch.cat([keys,~keys.any(1,keepdim=True)],1)
                bias=-F.softplus(self.align_strength[m])*self.distance/self.cfg.get('alignment_distance_scale',5.)
                h=self.cross[m](hs[m],keys,kv,bias)
            interacted.append(h)
        recon=[self.decoder[m](h) for m,h in enumerate(interacted)]
        alpha=[];zs=[];ym=[];reliability=[];recon_err=[]
        rho=1-obs.sum(-1).float()/lengths[:,None].clamp_min(1)
        for m,h in enumerate(interacted):
            a=masked_softmax(self.pool[m](h).squeeze(-1),obs[:,m]);alpha.append(a)
            z=(h*a[:,:,None]).sum(1)
            z=torch.where(obs[:,m].any(1,keepdim=True),z,self.absent[m]);zs.append(z)
            amplitude=3. if self.variant=='A9' else self.cfg.get('branch_output_bound',2.7)
            ym.append(amplitude*torch.tanh(self.head[m](z).squeeze(-1)))
            missing=valid & ~obs[:,m]
            starts=missing & ~F.pad(missing[:,:-1],(1,0),value=False)
            seg=starts.sum(1).float()/lengths.clamp_min(1)
            err=((recon[m].float()-x[m].float()).square().mean(-1)*obs[:,m]).sum(-1)/obs[:,m].sum(-1).clamp_min(1)
            recon_err.append(err)
            limit=self.cfg.get('reconstruction_residual_clip',25.)
            r=torch.stack([rho[:,m],seg,err.detach().clamp(max=limit)/limit],-1)
            if self.variant=='A5':r=r*0
            reliability.append(r)
        z=torch.stack(zs,1);ym=torch.stack(ym,1);present=obs.any(-1)
        logits=torch.cat([self.gate[m](torch.cat([z[:,m],reliability[m]],-1)) for m in range(3)],-1)
        w=masked_softmax(logits,present)
        if self.variant in ('A4','B_LMF','B_ConcatClean'):w=present.float()/present.sum(1,keepdim=True).clamp_min(1)
        fused=(z*w[:,:,None]).sum(1)
        yi=self.cfg.get('interaction_output_bound',.3)*torch.tanh(self.interaction(fused).squeeze(-1))
        if self.variant=='A9':yi=yi*0
        y=(w*ym).sum(1)+yi
        if self.variant in ('A4','B_ConcatClean'):y=3*torch.tanh(self.concat(z.flatten(1)).squeeze(-1))
        if self.variant=='B_LMF':
            f=[]
            for m in range(3):
                zz=torch.cat([torch.ones_like(z[:,m,:1]),z[:,m]],-1)
                f.append(torch.einsum('bd,drh->brh',zz,self.factors[m]))
            y=3*torch.tanh(self.lmf_head((f[0]*f[1]*f[2]).sum(1)).squeeze(-1))
        empty=~present.any(1)
        prior=3*torch.tanh(self.empty)
        y=torch.where(empty,prior,y);yi=torch.where(empty,prior,yi)
        return {'y':y,'logits':self.classifier(fused),'w':w,'y_m':ym,'alpha':torch.stack(alpha,1),
                'recon_err':torch.stack(recon_err,1),'y_int':yi,'z':fused,'recon':recon,
                'rho':rho,'obs':obs,'contributions':w*ym,'text_target':x[0]}
