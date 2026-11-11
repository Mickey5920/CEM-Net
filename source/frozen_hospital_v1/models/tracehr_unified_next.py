"""N8--N15: one jointly trained modular predictor per endpoint."""
import copy
import hashlib
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F
from models.tracehr_unified import UnifiedNetwork, independent_init


class EndpointResidual(UnifiedNetwork):
    def __init__(self, input_dim, prep, variant, endpoint, seed=42, width=256, depth=3):
        assert variant in ('N8', 'N9', 'N10', 'N11', 'N12') and endpoint in (0, 1)
        dropout = .1 if variant in ('N8', 'N9') else .2
        super().__init__(input_dim, prep, 'N1' if variant == 'N8' else 'N3', seed,
                         width=width, depth=depth, dropout=dropout)
        self.variant = variant
        self.heads = nn.ModuleList([self.heads[endpoint]])
        if variant in ('N11', 'N12'):
            with torch.no_grad():
                for module in self.modules():
                    if getattr(module, 'fast_r', None) is not None:
                        module.fast_r.fill_(1.)
                        module.fast_s.fill_(1.)
                if variant == 'N11':
                    with independent_init(seed, 'conservative/first_hidden'):
                        self.blocks[0].up.fast_r.normal_()
        if variant == 'N12':
            with independent_init(seed, 'conservative/input_features'):
                self.input_members = nn.Parameter(torch.randn(self.members, input_dim))

    def forward(self, raw):
        if self.variant != 'N12':
            return super().forward(raw)
        x = self.transform(raw)
        numeric = x.index_select(1, self.numeric_indices) * self.numerical.scale
        x = x.index_copy(1, self.numeric_indices, numeric)
        h = self.input_projection(x[:, None] * self.input_members[None])
        z = torch.cos(2 * math.pi * numeric.unsqueeze(-1) * self.numerical.frequency + self.numerical.phase)
        z = torch.bmm(z.transpose(0, 1), self.numerical.weight).transpose(0, 1) + self.numerical.bias
        # One coefficient is shared by each feature's raw and learned coordinates.
        scale = self.input_members.index_select(1, self.numeric_indices)
        z = (z[:, None] * scale[None, :, :, None]).flatten(2)
        h = self.input_dropout(F.gelu(self.input_norm((h + self.numerical.projection(z)) / math.sqrt(2))))
        for block in self.blocks:
            h = block(h)
        return self.heads[0](self.output_norm(h))

    def loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits[..., 0], y[:, None].expand(-1, self.members))

    def probabilities(self, raw):
        return self(raw).sigmoid().mean(1)


def fixed_to_buffers(module):
    names = []
    for prefix, child in module.named_modules():
        for name, value in list(child._parameters.items()):
            if value is not None and not value.requires_grad:
                del child._parameters[name]
                child.register_buffer(name, value.detach())
                names.append(prefix + '.' + name)
    return names


def fresh_native(raw, y, config, seed):
    sys.modules['lightning'] = None
    from pytabkit.models.training.nn_creator import NNCreator
    from pytabkit.models.data.data import DictDataset, TensorInfo
    from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources
    cfg = copy.deepcopy(config)
    for key in ('tmp_folder', 'device', 'random_state', 'val_metric_name'):
        cfg.pop(key, None)
    cfg.update(allow_single_split_opt=False, init_ram_limit_gb=1.)
    assert cfg['weight_param'] == 'ntk' and cfg['lr_sched'] == 'coslog4'
    assert cfg['n_epochs'] == 256 and not cfg['use_ls']
    assert cfg.get('l1_reg', 0) == cfg.get('l2_reg', 0) == 0
    n = len(raw)
    ds = DictDataset(dict(x_cont=torch.as_tensor(raw), x_cat=torch.empty((n, 0), dtype=torch.long),
                          y=torch.as_tensor(y[:, None], dtype=torch.long)),
                     dict(x_cont=TensorInfo(feat_shape=[raw.shape[1]]), x_cat=TensorInfo(cat_sizes=[]),
                          y=TensorInfo(cat_sizes=[2])))
    splits = [SplitIdxs(torch.arange(n)[None], None, None, seed, [seed], 0)]
    creator = NNCreator(**cfg)
    creator.setup_from_dataset(ds, splits, InterfaceResources(4, []))
    net = creator.create_model(ds, splits)
    fixed = fixed_to_buffers(net) + ['static.' + n for n in fixed_to_buffers(creator.static_model)]
    return net, creator.static_model, creator.hp_manager, dict(config=cfg, fixed_statistics=fixed)


class NativeMemberWeight(nn.Module):
    """Share a native NTK weight, preserving its context, scaling and optimizer."""
    def __init__(self, inner, members, first, feature_map, seed, name):
        super().__init__()
        self.inner, self.members = inner, members
        self.register_buffer('feature_map', feature_map if first else torch.arange(inner.weight.shape[-2]))
        if members == 1:
            self.register_parameter('fast_r', None)
            self.register_parameter('fast_s', None)
            return
        from pytabkit.models.nn_models.base import Variable
        dim = int(self.feature_map.max()) + 1
        with independent_init(seed, 'native_members/' + name):
            r = torch.randn(members, 1, dim) if first else torch.ones(members, 1, dim)
        with inner.weight.set_context():
            self.fast_r = Variable(r, hyper_factors={'lr': 1., 'wd': 1.})
            self.fast_s = Variable(torch.ones(members, 1, inner.weight.shape[-1]),
                                   hyper_factors={'lr': 1., 'wd': 1.})

    def forward(self, tensors):
        if self.members == 1:
            return self.inner(tensors)
        x = tensors['x_cont'] * self.fast_r.index_select(-1, self.feature_map)
        value = self.inner(dict(tensors, x_cont=x))
        return dict(value, x_cont=value['x_cont'] * self.fast_s)


class NativeModular(nn.Module):
    def __init__(self, raw, y, variant, config, seed=42):
        super().__init__()
        assert variant in ('N13', 'N14', 'N15')
        self.variant, self.members = variant, {'N13': 1, 'N14': 8, 'N15': 32}[variant]
        with torch.random.fork_rng(devices=[]):
            self.net, self.static, self.hp, self.info = fresh_native(raw, y, config, seed)
        layers = [(n, m) for n, m in self.net.named_modules() if type(m).__name__ == 'WeightLayer']
        embeddings = [m for m in self.net.modules() if type(m).__name__ == 'PLREmbeddingsLayerCosBias']
        assert len(layers) == 4 and len(embeddings) == 1
        embedding = embeddings[0]
        assert embedding.plr_use_densenet
        n, learned = embedding.weight_1.shape[-3], embedding.weight_2.shape[-1]
        # Native PBLD emits [all learned feature coordinates, all raw coordinates].
        fmap = torch.cat([torch.arange(n).repeat_interleave(learned), torch.arange(n)])
        assert fmap.numel() == layers[0][1].weight.shape[-2]
        self.info.update(embedding_features=n, learned_coordinates=learned,
                         member_feature_map_sha256=hashlib.sha256(fmap.numpy().tobytes()).hexdigest(),
                         weight_layers=[name for name, _ in layers])
        with torch.no_grad():
            x = torch.as_tensor(raw[:8].copy())
            t = self.static(dict(x_cont=x[None], x_cat=torch.empty((1, len(x), 0), dtype=torch.long)))
            expected = self.net(t)['x_cont'].clone()
        for i, (name, layer) in enumerate(layers):
            parent_name, leaf = name.rsplit('.', 1)
            parent = self.net.get_submodule(parent_name)
            parent._modules[leaf] = NativeMemberWeight(layer, self.members, i == 0, fmap, seed, name)
        if self.members == 1:
            with torch.no_grad():
                drift = float((self.net(t)['x_cont'] - expected).abs().max())
            assert drift == 0, ('Native wrapper changes initial function', drift)
            self.info['native_wrapper_initial_max_drift'] = drift
        assert all(p.requires_grad and hasattr(p, 'context') for p in self.parameters())

    def forward(self, raw):
        x = self.static(dict(x_cont=raw[None], x_cat=torch.empty((1, len(raw), 0), device=raw.device, dtype=torch.long)))
        return self.net(x)['x_cont'].transpose(0, 1)

    def loss(self, logits, y):
        return F.cross_entropy(logits.reshape(-1, 2), y.long()[:, None].expand(-1, self.members).reshape(-1))

    def probabilities(self, raw):
        return self(raw).softmax(-1)[..., 1].mean(1, keepdim=True)


def parameter_hash(model, common=False):
    value = hashlib.sha256()
    for name, p in model.named_parameters():
        if common and ('.fast_' in name or name == 'input_members'):
            continue
        value.update(name.encode())
        value.update(p.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()
