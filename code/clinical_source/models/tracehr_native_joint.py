"""Fresh native A/V/R prediction paths with bounded memberwise interactions."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.clinical_tracehr_fullstream import CachedC
from models.tracehr_joint_redesign import JointRedesign, make_realmlp
from models.tracehr_joint_recovery import MemberLinear
from models.tracehr_scratch_fusion import FreshA


class SingleA(CachedC):
    def forward(self, x):
        h = self.net(x)
        return h, self.head(h)


class BoundedInteraction(nn.Module):
    def __init__(self, k, a_dim, rank=16):
        super().__init__()
        self.k = k
        self.a_low = nn.Sequential(nn.Linear(a_dim, rank), nn.LayerNorm(rank))
        self.r_low = nn.Sequential(nn.Linear(256, rank), nn.LayerNorm(rank))
        self.v_low = nn.Sequential(nn.Linear(256, rank), nn.LayerNorm(rank))
        self.interaction = MemberLinear(k, 3*rank)
        self.eta_logit = nn.Parameter(torch.tensor(math.log(.1/.9)))

    def forward(self, a, r, v):
        aa, rr, vv = self.a_low(a.flatten(1))[:, None], self.r_low(r)[:, None], self.v_low(v)
        pair = torch.cat([aa*vv, rr*vv, (aa*rr).expand(-1, self.k, -1)], -1)
        return self.eta_logit.sigmoid()*self.interaction(pair).tanh()


def observation_summary(raw, phi_dim):
    # Existing simple_summary: observed, log1p(count), first/24, recency/24, ...
    simple = raw[:, phi_dim:phi_dim+352].reshape(-1, 32, 11)
    seen = simple[..., 0]
    count = seen.sum(1).clamp_min(1)
    return torch.stack([seen.mean(1), simple[..., 1].mean(1)/5.,
                        (simple[..., 2]*seen).sum(1)/count,
                        (simple[..., 3]*seen).sum(1)/count], -1)


def mixture_logit(component_logits, log_weights):
    """Both classes stay in log space; no clipping of extreme probabilities."""
    lp = torch.logsumexp(log_weights+F.logsigmoid(component_logits), -1)
    lq = torch.logsumexp(log_weights+F.logsigmoid(-component_logits), -1)
    return lp-lq


class NativeFusion(nn.Module):
    def __init__(self, kind, pos_weight, k=32, seed=42):
        super().__init__()
        assert kind in ('P0','P1','P2','P3','P4','P5') and k == 32
        self.kind, self.k = kind, k
        self.register_buffer('log_pos_weight', torch.as_tensor(pos_weight).detach().cpu().log())
        if kind != 'P0':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+5000)
                self.tasks = nn.ModuleList([BoundedInteraction(k, 96 if kind=='P5' else 192) for _ in range(2)])
        if kind == 'P4':
            # A separate RNG stream preserves the interaction initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+6000)
                self.gates = nn.ModuleList([nn.Sequential(nn.Linear(11,32),nn.GELU(),nn.Linear(32,3)) for _ in range(2)])
                for gate in self.gates:
                    nn.init.normal_(gate[-1].weight, std=.01)
                    nn.init.normal_(gate[-1].bias, std=.001)

    def forward(self, a, r, v, aux, observations=None):
        az = aux['A']-self.log_pos_weight
        vz = aux['V']
        rz = aux['R'][..., 1]-aux['R'][..., 0]
        assert vz.shape[1:] == (self.k, 2)
        components = torch.stack([az[:,None].expand_as(vz), vz, rz[:,None].expand_as(vz)], -1)
        if self.kind == 'P4':
            assert observations is not None and observations.shape[1] == 4
            pv = vz.sigmoid()
            scores = torch.stack([az.sigmoid(),pv.mean(1),rz.sigmoid(),
                (az/5.).tanh(),(vz.mean(1)/5.).tanh(),(rz/5.).tanh(),pv.var(1,unbiased=False)], -1)
            inp = torch.cat([scores,observations[:,None].expand(-1,2,-1)], -1)
            weights = torch.stack([gate(inp[:,t]) for t,gate in enumerate(self.gates)],1).log_softmax(-1)
            weights = weights[:,None]
        else:
            weights = torch.full_like(components[:, :1], -math.log(3.))
        base = mixture_logit(components, weights)
        delta = torch.zeros_like(base) if self.kind=='P0' else torch.stack(
            [head(a,r[:,t],v[:,t]) for t,head in enumerate(self.tasks)], -1)
        # Detached values are diagnostics only; neither base nor delta is detached in the predictor.
        self.last = dict(base=base.detach(),delta=delta.detach(),weights=weights.detach().exp(),
                         a_calibrated=az.detach().sigmoid())
        return base+delta


class NativeJointModel(JointRedesign):
    def __init__(self, data, branches, fusion, seed, r_config, v_config):
        nn.Module.__init__(self)
        assert branches == 'AVR'
        self.branches, self.kind, self.recipe, self.phi_dim = branches, fusion, 'native', data.phi_dim
        self.initialization, self.hp_managers = {}, []
        for e in branches:
            data.initialization_progress(e)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+1000*('AVR'.index(e)+1))
                if e == 'A':
                    self.a = SingleA(data.phi_dim) if fusion=='P5' else FreshA(data.phi_dim,58 if data.database=='mimic' else 53)
                elif e == 'V':
                    from tabm import TabM
                    self.v = nn.ModuleList([TabM.make(n_num_features=data.input_dim,d_out=1,k=v_config['k'],
                        d_block=v_config['width'],n_blocks=v_config['depth'],dropout=v_config['dropout']) for _ in range(2)])
                else:
                    self.r, self.r_static = nn.ModuleList(), nn.ModuleList()
                    raw = data.initialization_raw()
                    for t in range(2):
                        net,static,hp,info = make_realmlp(raw,data.targets['train']['y'][:,t],seed+3000+t,r_config,True)
                        self.r.append(net);self.r_static.append(static);self.hp_managers.append(hp)
                        self.initialization[f'R{t}'] = info
                    del raw
        self.fusion = NativeFusion(fusion,data.pos_weight,v_config['k'],seed)
        assert all(p.requires_grad for p in self.parameters())

    def forward(self, batch):
        raw, n = batch['raw'], len(batch['raw'])
        a,az = self.a(raw[:, :self.phi_dim])
        vh,vz,rh,rz = [],[],[],[]
        for net in self.v:
            found=[]
            hook=net.output.register_forward_pre_hook(lambda module,args:found.append(args[0]))
            try:pred=net(batch['v'])
            finally:hook.remove()
            vh.append(found[0]);vz.append(pred.squeeze(-1))
        for t,(net,static) in enumerate(zip(self.r,self.r_static)):
            x=static(dict(x_cont=raw[None],x_cat=torch.empty((1,n,0),device=raw.device,dtype=torch.long)))
            found=[]
            hook=net.get_submodule(self.initialization[f'R{t}']['prehead_module']).register_forward_pre_hook(
                lambda module,args:found.append(args[0]['x_cont']))
            try:pred=net(x)['x_cont']
            finally:hook.remove()
            rh.append(found[0][0]);rz.append(pred[0])
        aux=dict(A=az,V=torch.stack(vz,-1),R=torch.stack(rz,1))
        obs=observation_summary(raw,self.phi_dim) if self.kind=='P4' else None
        z=self.fusion(a,torch.stack(rh,1),torch.stack(vh,1),aux,obs)
        return z,aux


def native_joint_loss(logits, y, kind):
    if kind == 'P3':
        lp=torch.logsumexp(F.logsigmoid(logits),1)-math.log(logits.shape[1])
        lq=torch.logsumexp(F.logsigmoid(-logits),1)-math.log(logits.shape[1])
        return -(y*lp+(1-y)*lq).mean()
    return F.binary_cross_entropy_with_logits(logits,y[:,None].expand_as(logits))
