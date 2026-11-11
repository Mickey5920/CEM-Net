"""Randomly initialized recovery controls and direct-path member fusion."""
import math
import torch
from torch import nn
from models.tracehr_joint_redesign import JointRedesign, make_realmlp, SeparateRead
from models.tracehr_scratch_fusion import FreshA


class MemberLinear(nn.Module):
    def __init__(self, k, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, width))
        self.bias = nn.Parameter(torch.empty(k))
        nn.init.uniform_(self.weight, -width**-.5, width**-.5)
        nn.init.uniform_(self.bias, -width**-.5, width**-.5)

    def forward(self, x):
        return (x*self.weight[None]).sum(-1)+self.bias[None]


class DirectTaskFusion(nn.Module):
    def __init__(self, kind, k=32, width=128, rank=16):
        super().__init__()
        assert kind in ('H0', 'H1', 'H2')
        self.kind, self.k = kind, k
        self.a_direct = nn.Linear(192, 1, bias=False)
        self.r_direct = nn.Linear(256, 1, bias=False)
        self.v_direct = MemberLinear(k, 256)
        if kind != 'H0':
            self.eta_logit = nn.Parameter(torch.tensor(math.log(.1/.9)))
        if kind == 'H1':
            self.a_low = nn.Sequential(nn.Linear(192, rank), nn.LayerNorm(rank))
            self.r_low = nn.Sequential(nn.Linear(256, rank), nn.LayerNorm(rank))
            self.v_low = nn.Sequential(nn.Linear(256, rank), nn.LayerNorm(rank))
            self.interaction = MemberLinear(k, 3*rank)
        elif kind == 'H2':
            self.a_proj, self.r_proj, self.v_proj = nn.Linear(96, width), nn.Linear(256, width), nn.Linear(256, width)
            self.member = nn.Parameter(torch.randn(k, width)*.02)
            self.read_a = SeparateRead(width, .2)
            self.gate = nn.Linear(3*width, 2)
            self.interaction = MemberLinear(k, width)

    def forward(self, a, r, v):
        assert a.shape[1:] == (2, 96) and v.shape[1:] == (self.k, 256)
        af = a.flatten(1)
        z = self.a_direct(af)+self.r_direct(r)+self.v_direct(v)
        if self.kind == 'H0':
            return z
        if self.kind == 'H1':
            al, rl, vl = self.a_low(af)[:, None], self.r_low(r)[:, None], self.v_low(v)
            pair = torch.cat([al*vl, rl*vl, (al*rl).expand(-1, self.k, -1)], -1)
            delta = self.interaction(pair)
        else:
            q = self.v_proj(v)+self.member[None]
            ca = self.read_a(q, self.a_proj(a))
            # A single R token has attention weight one; avoid redundant Q/K parameters.
            cr = self.r_proj(r)[:, None].expand(-1, self.k, -1)
            gate = self.gate(torch.cat([q, ca, cr], -1)).softmax(-1)
            delta = self.interaction(gate[..., :1]*ca+gate[..., 1:]*cr)
        return z+self.eta_logit.sigmoid()*delta


class DirectMemberFusion(nn.Module):
    def __init__(self, kind, k=32):
        super().__init__()
        self.tasks = nn.ModuleList([DirectTaskFusion(kind, k) for _ in range(2)])

    def forward(self, a, r, v):
        return torch.stack([head(a, r[:, t], v[:, t]) for t, head in enumerate(self.tasks)], -1)


class RecoveryModel(JointRedesign):
    def __init__(self, data, branches, fusion, seed, r_config, v_config):
        assert branches in ('A', 'V', 'R', 'AVR')
        if branches == 'AVR':
            super().__init__(data, 'F4l', 'native', seed, r_config, v_config)
            if fusion != 'F4l':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+5000)
                    self.fusion = DirectMemberFusion(fusion, v_config['k'])
            self.kind = fusion
            return
        nn.Module.__init__(self)
        assert fusion == 'solo'
        self.branches, self.kind, self.recipe, self.phi_dim = branches, fusion, 'native', data.phi_dim
        self.initialization, self.hp_managers = {}, []
        data.initialization_progress(branches)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1000*('AVR'.index(branches)+1))
            if branches == 'A':
                self.a = FreshA(data.phi_dim, 58 if data.database == 'mimic' else 53)
            elif branches == 'V':
                from tabm import TabM
                self.v = nn.ModuleList([TabM.make(n_num_features=data.input_dim, d_out=1, k=v_config['k'],
                    d_block=v_config['width'], n_blocks=v_config['depth'], dropout=v_config['dropout']) for _ in range(2)])
            else:
                self.r, self.r_static = nn.ModuleList(), nn.ModuleList()
                raw = data.initialization_raw()
                for t in range(2):
                    net, static, hp, info = make_realmlp(raw, data.targets['train']['y'][:, t], seed+3000+t, r_config, True)
                    self.r.append(net); self.r_static.append(static); self.hp_managers.append(hp)
                    self.initialization[f'R{t}'] = info
        assert all(p.requires_grad for p in self.parameters())

    def forward(self, batch):
        if self.branches == 'AVR':
            return super().forward(batch)
        if self.branches == 'A':
            _, z = self.a(batch['raw'][:, :self.phi_dim])
            return z, dict(A=z)
        if self.branches == 'V':
            z = torch.stack([net(batch['v']).squeeze(-1) for net in self.v], -1)
            return z, dict(V=z)
        raw = batch['raw']; n = len(raw); zz = []
        for net, static in zip(self.r, self.r_static):
            x = static(dict(x_cont=raw[None], x_cat=torch.empty((1, n, 0), device=raw.device, dtype=torch.long)))
            zz.append(net(x)['x_cont'][0])
        z = torch.stack(zz, 1)
        return z[..., 1]-z[..., 0], dict(R=z)
