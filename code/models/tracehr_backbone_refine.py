"""D0--D7: typed periodic input and one shared, jointly trained backbone."""
import copy
import hashlib
import math
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.tracehr_unified import independent_init
from models.tracehr_unified_next import fixed_to_buffers
from models.tracehr_native_transfer import TransferNetwork


class PostScaleResidual(TransferNetwork):
    def __init__(self, input_dim, prep, endpoint, seed=42, dropout=.2, width=256, depth=3):
        super().__init__(input_dim, prep, 'C4', endpoint, seed, width, depth)
        self.coordinate_scale = nn.Parameter(torch.ones(len(self.numeric_indices), 3))
        self.register_buffer('passthrough', torch.as_tensor(prep.get('passthrough', prep['binary']), dtype=torch.bool))
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    def transform(self, raw):
        z = (raw-self.center)*self.factor
        z = z/torch.sqrt(1+(z/3).square())
        return torch.where(self.passthrough, raw, z)

    def forward(self, raw):
        x = self.transform(raw)
        numeric = x.index_select(1, self.numeric_indices)
        z = torch.cos(2*math.pi*numeric.unsqueeze(-1)*self.numerical.frequency+self.numerical.phase)
        z = torch.bmm(z.transpose(0, 1), self.numerical.weight).transpose(0, 1)+self.numerical.bias
        x = x.index_copy(1, self.numeric_indices, numeric*self.numerical.scale)
        h = self.input_projection(x[:, None]*self.input_members[None])
        member_scale = self.input_members.index_select(1, self.numeric_indices)
        z = (z[:, None]*self.coordinate_scale[None, None]*member_scale[None, :, :, None]).flatten(2)
        h = self.input_dropout(F.gelu(self.input_norm((h+self.numerical.projection(z))/math.sqrt(2))))
        for block in self.blocks:
            h = block(h)
        return self.heads[0](self.output_norm(h))


class TypedPeriodic(nn.Module):
    """Keep the C4 numeric/binary mask and its exact periodic initialization."""
    def __init__(self, prep, seed):
        super().__init__()
        for name in ('center', 'factor', 'binary', 'numeric_indices'):
            dtype = torch.bool if name=='binary' else torch.long if name=='numeric_indices' else torch.float32
            self.register_buffer(name, torch.as_tensor(prep[name], dtype=dtype))
        self.register_buffer('passthrough', torch.as_tensor(prep.get('passthrough', prep['binary']), dtype=torch.bool))
        n = len(self.numeric_indices)
        with independent_init(seed, 'numerical/periodic'):
            self.frequency = nn.Parameter(.1*torch.randn(n, 16))
            self.phase = nn.Parameter(math.pi*(2*torch.rand(n, 16)-1))
            self.weight = nn.Parameter((2*torch.rand(n, 16, 3)-1)/4)
            self.bias = nn.Parameter((2*torch.rand(n, 3)-1)/4)

    def forward(self, raw):
        x = (raw-self.center)*self.factor
        x = torch.where(self.passthrough, raw, x/torch.sqrt(1+(x/3).square()))
        numeric = x.index_select(1, self.numeric_indices)
        z = torch.cos(2*math.pi*numeric.unsqueeze(-1)*self.frequency+self.phase)
        z = torch.bmm(z.transpose(0, 1), self.weight).transpose(0, 1)+self.bias
        return torch.cat((z.flatten(1), x), dim=1)


def native_trunk(encoded, y, config, seed):
    sys.modules['lightning'] = None
    from pytabkit.models.training.nn_creator import NNCreator
    from pytabkit.models.data.data import DictDataset, TensorInfo
    from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources
    cfg = copy.deepcopy(config)
    for key in ('tmp_folder', 'device', 'random_state', 'val_metric_name'):
        cfg.pop(key, None)
    cfg.update(tfms=[], num_emb_type='none', add_front_scale=True,
               allow_single_split_opt=False, init_ram_limit_gb=1., n_epochs=256, use_ls=False)
    assert cfg['weight_param']=='ntk' and cfg['lr_sched']=='coslog4'
    assert cfg.get('l1_reg', 0)==cfg.get('l2_reg', 0)==0
    n = len(encoded)
    ds = DictDataset(dict(x_cont=encoded, x_cat=torch.empty((n, 0), dtype=torch.long),
                          y=torch.as_tensor(y[:, None], dtype=torch.long)),
                     dict(x_cont=TensorInfo(feat_shape=[encoded.shape[1]]), x_cat=TensorInfo(cat_sizes=[]),
                          y=TensorInfo(cat_sizes=[2])))
    sub_seed = int(np.random.RandomState(seed).randint(0, 2**31-1))
    splits = [SplitIdxs(torch.arange(n)[None], None, None, seed, [sub_seed], 0)]
    creator = NNCreator(**cfg)
    creator.setup_from_dataset(ds, splits, InterfaceResources(4, []))
    net = creator.create_model(ds, splits)
    fixed = fixed_to_buffers(net)+['static.'+n for n in fixed_to_buffers(creator.static_model)]
    return net, creator.static_model, creator.hp_manager, dict(config=cfg, fixed_statistics=fixed,
        split_seed=seed, sub_split_seed=sub_seed, adaptation='typed periodic frontend; official native trunk and optimizer')


class TypedNative(nn.Module):
    def __init__(self, raw, y, prep, config, members, seed=42, init_rows=1024):
        super().__init__()
        assert members in (1, 4, 8)
        self.members = members
        self.numerical = TypedPeriodic(prep, seed)
        ix = np.random.RandomState(seed).permutation(len(raw))[:min(init_rows, len(raw))]
        # The official data-dependent RANDOM initializer sees train rows only.
        # No trained weights or previous model predictions enter this fit.
        with torch.no_grad():
            encoded = torch.cat([self.numerical(torch.as_tensor(np.array(raw[i], dtype=np.float32)))
                                 for i in np.array_split(ix, max(1, math.ceil(len(ix)/64)))])
        with torch.random.fork_rng(devices=[]):
            self.net, self.static, self.hp, self.info = native_trunk(encoded, y[ix], config, seed)
        del encoded
        layers = [m for m in self.net.modules() if type(m).__name__=='WeightLayer']
        assert len(layers)==4 and any(type(m).__name__=='ScaleLayer' for m in self.net.modules())
        from pytabkit.models.nn_models.base import Variable
        # Attach native optimizer metadata to the four typed-embedding tensors.
        for name, p in list(self.numerical.named_parameters()):
            with layers[0].weight.set_context():
                setattr(self.numerical, name, Variable(p.detach(), hyper_factors={'lr': .1, 'wd': 1.}))
        fmap = torch.cat((self.numerical.numeric_indices.repeat_interleave(3), torch.arange(raw.shape[1])))
        self.register_buffer('feature_map', fmap)
        assert len(fmap)==layers[0].weight.shape[-2]
        if members>1:
            with independent_init(seed, 'conservative/input_features'):
                self.input_members = nn.Parameter(torch.randn(8, raw.shape[1])[:members].clone())
        else:
            self.register_parameter('input_members', None)
        self.info.update(initialization_rows=len(ix), initialization_split='train',
                         initialization_indices_sha256=hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest(),
                         numeric_features=len(self.numerical.numeric_indices), learned_coordinates=3,
                         post_encoding_scale=True, shared_weight_layers=4, hidden_member_adapters=0,
                         independent_member_heads=0, members=members)
        assert all(p.requires_grad for p in self.parameters())

    def forward(self, raw):
        x = self.numerical(raw)[None]
        if self.input_members is not None:
            x = x*self.input_members.index_select(1, self.feature_map)[:, None]
        tensors = dict(x_cont=x, x_cat=torch.empty((self.members, len(raw), 0), dtype=torch.long, device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0, 1)

    def loss(self, logits, y):
        return F.cross_entropy(logits.reshape(-1, 2), y.long()[:, None].expand(-1, self.members).reshape(-1))

    def probabilities(self, raw):
        return self(raw).softmax(-1)[..., 1].mean(1, keepdim=True)
