"""Fresh A/V/R encoders, faithful RealMLP runtime, and member-preserving fusion."""
import copy
import sys
import torch
from torch import nn
from torch.nn import functional as F
from models.tracehr_scratch_fusion import FreshA, fixed_statistics_to_buffers, realmlp_prediction_layer
from models.tracehr_module_fusion import ModuleFusion, ResidualReadout


def make_realmlp(raw, y, seed, config, restored):
    sys.modules['lightning'] = None
    from pytabkit.models.training.nn_creator import NNCreator
    from pytabkit.models.data.data import DictDataset, TensorInfo
    from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources
    cfg = copy.deepcopy(config)
    for key in ('tmp_folder', 'device', 'random_state', 'val_metric_name'):
        cfg.pop(key, None)
    cfg.update(allow_single_split_opt=False, init_ram_limit_gb=1.)
    if not restored:
        cfg.update(p_drop_sched='constant', wd_sched='constant', lr_sched='constant')
    assert not cfg.get('use_ls', False)
    assert cfg.get('l1_reg', 0) == cfg.get('l2_reg', 0) == 0
    tensors = dict(x_cont=torch.as_tensor(raw), x_cat=torch.empty((len(raw), 0), dtype=torch.long),
                   y=torch.as_tensor(y[:, None], dtype=torch.long))
    ds = DictDataset(tensors, dict(x_cont=TensorInfo(feat_shape=[raw.shape[1]]),
                     x_cat=TensorInfo(cat_sizes=[]), y=TensorInfo(cat_sizes=[2])))
    splits = [SplitIdxs(torch.arange(len(raw))[None], None, None, seed, [seed], 0)]
    creator = NNCreator(**cfg)
    creator.setup_from_dataset(ds, splits, InterfaceResources(4, []))
    net = creator.create_model(ds, splits)
    fixed = fixed_statistics_to_buffers(net)
    fixed += ['static.' + s for s in fixed_statistics_to_buffers(creator.static_model)]
    info = dict(config=cfg, fixed_statistics=fixed, prehead_module=realmlp_prediction_layer(net))
    return net, creator.static_model, creator.hp_manager, info


class SeparateRead(nn.Module):
    """Normalize attention separately inside each module; retain value amplitudes."""
    def __init__(self, width, dropout):
        super().__init__()
        self.qn, self.kn = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)

    def forward(self, q, memory):
        return self.attn(self.qn(q), self.kn(memory), memory, need_weights=False)[0]


class TaskFusion(nn.Module):
    def __init__(self, kind, k=32, width=128, dropout=.2):
        super().__init__()
        self.kind, self.k, self.width = kind, k, width
        self.a = nn.Linear(96, width)
        self.r = nn.Linear(256, width)
        self.v = nn.Linear(256, width)
        self.alpha = nn.Parameter(torch.tensor(.1))
        self.head = ResidualReadout(width, 256, dropout)
        if kind == 'F0m':
            self.mix = nn.Sequential(nn.Linear(3*width, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, width))
        elif kind in ('F1m', 'F2p', 'F3b'):
            self.r_tokens = nn.Linear(width, 4*width)
            self.member = nn.Parameter(torch.randn(k, width)*.02)
            self.read_a, self.read_r = SeparateRead(width, dropout), SeparateRead(width, dropout)
            if kind == 'F1m':
                self.mix = nn.Linear(2*width, width)
            elif kind == 'F2p':
                self.read_ar = SeparateRead(width, dropout)
                self.mix = nn.Sequential(nn.Linear(3*width, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, width))
            else:
                self.bridge = nn.Parameter(torch.randn(4, width)*.02)
                self.read_v, self.read_bridge = SeparateRead(width, dropout), SeparateRead(width, dropout)
                self.mix = nn.Linear(3*width, width)
        elif kind == 'F4l':
            self.low = nn.ModuleDict({x: nn.Linear(width, 16) for x in 'arv'})
            self.mix = nn.Linear(48, width)
        else:
            raise ValueError(kind)

    def forward(self, a, r, v):
        # A: [B,2,96], R: [B,256], V: [B,K,256]. This instance belongs to one task.
        aa, rr, vv = self.a(a), self.r(r), self.v(v)
        ag = aa.mean(1)[:, None].expand(-1, self.k, -1)
        rg = rr[:, None].expand(-1, self.k, -1)
        if self.kind == 'F0m':
            delta = self.mix(torch.cat([vv, ag, rg], -1))
        elif self.kind == 'F4l':
            x, y, z = self.low['a'](ag), self.low['r'](rg), self.low['v'](vv)
            delta = self.mix(torch.cat([x*y, x*z, y*z], -1))
        else:
            memory_r = self.r_tokens(rr).reshape(len(r), 4, self.width)
            q = vv + self.member[None]
            if self.kind == 'F3b':
                b = self.bridge[None].expand(len(r), -1, -1)
                b = b + self.mix(torch.cat([self.read_a(b, aa), self.read_r(b, memory_r), self.read_v(b, vv)], -1))
                delta = self.read_bridge(q, b)
            else:
                values = [self.read_a(q, aa), self.read_r(q, memory_r)]
                if self.kind == 'F2p':
                    # A-R shared context, A-V and R-V member-specific reads.
                    ar = self.read_ar(aa, memory_r).mean(1)[:, None].expand(-1, self.k, -1)
                    values.append(ar)
                delta = self.mix(torch.cat(values, -1))
        return self.head(vv + self.alpha*delta)


class MemberFusion(nn.Module):
    def __init__(self, kind, k=32):
        super().__init__()
        self.tasks = nn.ModuleList([TaskFusion(kind, k) for _ in range(2)])

    def forward(self, a, r, v):
        assert v.ndim == 4 and v.shape[1] == 2 and r.shape[1] == 2
        return torch.stack([net(a, r[:, t], v[:, t]) for t, net in enumerate(self.tasks)], -1)


class JointRedesign(nn.Module):
    def __init__(self, data, fusion, recipe, seed, r_config, v_config):
        super().__init__()
        self.branches, self.phi_dim = 'AVR', data.phi_dim
        self.kind, self.recipe = fusion, recipe
        self.initialization, self.hp_managers = {}, []
        for e in self.branches:
            if hasattr(data, 'initialization_progress'):
                data.initialization_progress(e)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1000*('AVR'.index(e)+1))
                if e == 'A':
                    self.a = FreshA(data.phi_dim, 58 if data.database == 'mimic' else 53)
                elif e == 'V':
                    from tabm import TabM
                    self.v = nn.ModuleList([TabM.make(n_num_features=data.input_dim, d_out=1, k=v_config['k'],
                        d_block=v_config['width'], n_blocks=v_config['depth'], dropout=v_config['dropout']) for _ in range(2)])
                else:
                    self.r, self.r_static = nn.ModuleList(), nn.ModuleList()
                    raw = data.initialization_raw()
                    for t in range(2):
                        net, static, hp, info = make_realmlp(raw, data.targets['train']['y'][:, t], seed+3000+t, r_config, recipe == 'native')
                        self.r.append(net); self.r_static.append(static); self.hp_managers.append(hp)
                        self.initialization[f'R{t}'] = info
                    del raw
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+5000)
            self.fusion = ModuleFusion('AVR', 'F0') if fusion == 'F0' else MemberFusion(fusion, v_config['k'])
        assert all(p.requires_grad for p in self.parameters())

    def forward(self, batch):
        raw, n = batch['raw'], len(batch['raw'])
        a, az = self.a(raw[:, :self.phi_dim])
        vh, vz, rh, rz = [], [], [], []
        for net in self.v:
            found = []
            hook = net.output.register_forward_pre_hook(lambda module, args: found.append(args[0]))
            try:
                pred = net(batch['v'])
            finally:
                hook.remove()
            vh.append(found[0]); vz.append(pred.squeeze(-1))
        for t, (net, static) in enumerate(zip(self.r, self.r_static)):
            x = static(dict(x_cont=raw[None], x_cat=torch.empty((1,n,0), device=raw.device, dtype=torch.long)))
            found = []
            hook = net.get_submodule(self.initialization[f'R{t}']['prehead_module']).register_forward_pre_hook(
                lambda module, args: found.append(args[0]['x_cont']))
            try:
                pred = net(x)['x_cont']
            finally:
                hook.remove()
            rh.append(found[0][0]); rz.append(pred[0])
        v, r = torch.stack(vh, 1), torch.stack(rh, 1)
        if self.kind == 'F0':
            hidden = dict(A=a, V=v.flatten(1,2), R=r)
            masks = {e: torch.ones(h.shape[:2], device=raw.device, dtype=torch.bool) for e,h in hidden.items()}
            z = self.fusion(hidden, masks)
        else:
            z = self.fusion(a, r, v)
        return z, dict(A=az, V=torch.stack(vz, -1), R=torch.stack(rz, 1))


def joint_probability(logits):
    return logits.sigmoid().mean(1) if logits.ndim == 3 else logits.sigmoid()


def joint_loss(logits, y):
    if logits.ndim == 3:
        y = y[:, None].expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, y)
