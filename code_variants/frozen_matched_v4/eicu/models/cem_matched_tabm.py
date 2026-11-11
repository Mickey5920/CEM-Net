"""Native TabM-mini predictor with the approved common response input and recipe.

The linear functions retain TabM's initialization. Their parameters are expressed
in the corresponding CEM NTK coordinates before inheriting its rolewise optimizer.
This avoids applying an unscaled 0.04 learning rate to standard linear weights.
"""
import copy
import torch
from torch import nn
from models.tracehr_unified import independent_init


class MatchedLinear(nn.Module):
    def __init__(self, original, reference_weight, reference_bias):
        super().__init__()
        from pytabkit.models.nn_models.base import Variable
        self.ensemble = original.weight.ndim == 3
        self.factor, self.bias_factor = float(reference_weight.factor), float(reference_bias.factor)
        w = original.weight.detach() if self.ensemble else original.weight.detach().T.unsqueeze(0)
        b = original.bias.detach() if self.ensemble else original.bias.detach().unsqueeze(0)
        with reference_weight.weight.set_context():
            self.weight = Variable(w.clone()/self.factor, hyper_factors=copy.deepcopy(reference_weight.weight.hyper_factors))
        with reference_bias.bias.set_context():
            self.bias = Variable(b.clone()/self.bias_factor, hyper_factors=copy.deepcopy(reference_bias.bias.hyper_factors))

    def forward(self, x):
        if self.ensemble:
            return ((x.transpose(0, 1) @ self.weight)*self.factor).transpose(0, 1)+self.bias*self.bias_factor
        return (x @ self.weight[0])*self.factor+self.bias*self.bias_factor


class MatchedTabM(nn.Module):
    def __init__(self, fresh_cem_base, seed, dropout):
        super().__init__()
        from tabm import TabM
        self.members = 8
        self.numerical = fresh_cem_base.numerical
        self.input_members = nn.Parameter(fresh_cem_base.input_members.detach().clone())
        self.hp = fresh_cem_base.hp
        self.info = copy.deepcopy(fresh_cem_base.info)
        width = self.input_members.shape[1]
        with independent_init(seed, 'matched/tabm-mini'):
            net = TabM.make(n_num_features=width, d_out=2, k=8, d_block=256,
                            n_blocks=3, dropout=dropout, arch_type='tabm-mini', activation='ReLU')
        weights = [x for x in fresh_cem_base.net.modules() if type(x).__name__ == 'WeightLayer']
        biases = [x for x in fresh_cem_base.net.modules() if type(x).__name__ == 'BiasLayer']
        assert len(weights) == len(biases) == 4
        assert net.backbone.affine.bias is None
        net.backbone.affine.weight.data.copy_(self.input_members)
        # Native head, native activation and native initial effective linear weights.
        net.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+21000)
            x = torch.randn(3, width)
            before = net(x).clone()
        mappings = []
        for i in range(3):
            layer = net.backbone.blocks[i][0]
            net.backbone.blocks[i][0] = MatchedLinear(layer, weights[i], biases[i])
            mappings.append(dict(role=f'hidden_{i}', factor=weights[i].factor,
                                 weight_factors=weights[i].weight.hyper_factors,
                                 bias_factors=biases[i].bias.hyper_factors))
        net.output = MatchedLinear(net.output, weights[-1], biases[-1])
        net.backbone.affine = nn.Identity()
        self.net = net
        with torch.no_grad():
            after = self.net(x[:, None]*self.input_members[None])
            torch.testing.assert_close(after, before, rtol=3e-5, atol=3e-6)
        self.info.update(matched_control=True, architecture='official TabM-mini 0.0.3',
                         native_activation='ReLU', independent_member_heads=8,
                         shared_weight_layers=3, linear_parameterization='CEM role-matched NTK',
                         effective_linear_initialization='native TabM',
                         input_factor_initialization='same fresh grouped normal values as CEM; independently learned coordinates',
                         linear_mapping=mappings, native_forward_max_abs=float((after-before).abs().max()),
                         front_scale=False, all_frontend_parameters_jointly_trained=True)

    def forward(self, raw):
        x = self.numerical(raw)
        return self.net(x[:, None]*self.input_members[None])

    def loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.long()[:, None].expand(-1, 8).flatten())


class Predictor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base, self.members = base, 8
        self.info = base.info
        self.auxiliary_enabled = self.main_residual_enabled = False

    def forward(self, x):
        return self.base(x['raw'])

    def loss(self, logits, labels):
        return .5*self.base.loss(logits, labels)
