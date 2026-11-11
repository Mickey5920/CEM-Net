"""Two prespecified gate controls; evidence channels and optimization are unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from models.cem_gated_event import GatedEventChannels
from models.cem_refine import RefineStructure
from models.tracehr_unified import independent_init
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('V1', 'V2')
DESCRIPTIONS = {
    'V1': 'input-independent learned scalar event gate',
    'V2': 'event gate additionally conditioned on the current member backbone hidden state',
}


def expected_new_names(variant):
    assert variant in VARIANTS
    return {'gate.bias'} if variant == 'V1' else {'gate.weight', 'gate.bias', 'backbone_gate.weight'}


def expected_removed_names(variant, original_names):
    assert variant in VARIANTS
    return set()


class ScalarGate(nn.Module):
    def __init__(self, original_bias):
        super().__init__()
        # Copy only a freshly constructed U1 bias, never a learned checkpoint.
        self.bias = nn.Parameter(original_bias.detach().clone())

    def forward(self, inputs):
        return self.bias.expand(*inputs.shape[:-1], 1)


class GateChannels(GatedEventChannels):
    def __init__(self, num_types, global_dim, variant, seed):
        assert variant in VARIANTS
        super().__init__(num_types, global_dim, seed)
        self.gate_variant = variant
        if variant == 'V1':
            self.gate = ScalarGate(self.gate.bias)
        else:
            with independent_init(seed, 'CEM/evidence/backbone_event_gate'):
                self.backbone_gate = nn.Linear(256, 1, bias=False)

    def encode(self, scaled, batch, phi_dim):
        # Same U1 evidence computation and stochastic-operation order. Only the
        # deterministic gate/readout is deferred until current hidden is available.
        ids, mask = batch['event_ids'], batch['event_mask']
        safe_ids = ids.clamp_max(self.num_types-1)
        values = torch.stack([scaled[:, i*self.num_types:(i+1)*self.num_types].gather(1, safe_ids)
                              for i in range(4)], -1)
        tokens = self.token_ffn(self.type_embedding(ids)+self.feature_projection(values))
        global_token = self.global_projection(scaled[:, 4*self.num_types:phi_dim])[:, None]
        tokens = torch.cat((global_token, tokens), 1)
        mask = torch.cat((torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask), 1)
        latent = self.latents[None].expand(len(tokens), -1, -1)
        read, _ = self.cross_attention(latent, tokens, tokens, key_padding_mask=~mask, need_weights=True)
        latent = self.latent_ffn(self.cross_norm(latent+read))
        read, _ = self.self_attention(latent, latent, latent, need_weights=True)
        latent = self.self_ffn(self.self_norm(latent+read))
        event = self.pool(latent)
        direct = self.residual_scale*self.output(self.output_norm(event))
        simple = scaled[:, phi_dim:phi_dim+352].reshape(-1, 32, 11)
        state = scaled[:, phi_dim+352:].reshape(-1, 32, 15)
        labs = self.lab_projection(torch.cat((simple, state), -1))+self.lab_identity.weight[None]
        labs = torch.cat((torch.zeros_like(labs[:, :1]), labs), 1)
        present = batch['raw'][:, phi_dim:phi_dim+352:11] > 0
        lab_mask = torch.cat((torch.ones_like(present[:, :1]), present), 1)
        read, _ = self.lab_attention(latent, labs, labs, key_padding_mask=~lab_mask, need_weights=True)
        cross = self.lab_ffn(self.lab_norm(latent+read))
        interaction = self.pool(cross)
        return event, interaction, direct

    def readout(self, event, interaction, direct, hidden):
        assert self.gate_variant == 'V2' and hidden.shape == (8, len(event), 256)
        gate_input = torch.cat((F.layer_norm(event, (self.width,)),
                                F.layer_norm(interaction, (self.width,))), -1)
        # Pre-residual current-member h. No member averaging or stop-gradient.
        h = F.layer_norm(hidden.transpose(0, 1), (256,))
        g = (self.gate(gate_input)+self.backbone_gate(h)).sigmoid()
        if not torch.is_grad_enabled():
            self.last_gate_summary = dict(scope='last_inference_batch_only', rows=len(g),
                mean=float(g.mean()), std=float(g.std(unbiased=False)), minimum=float(g.min()), maximum=float(g.max()))
        return (g*direct+self.extra_scale*self.extra_output(interaction)).transpose(0, 1)

    def forward(self, scaled, batch, phi_dim, hidden=None):
        if self.gate_variant == 'V1':
            return super().forward(scaled, batch, phi_dim)
        return self.readout(*self.encode(scaled, batch, phi_dim), hidden=hidden)


class GateStructure(RefineStructure):
    def __init__(self, fresh_cem, phi_dim, num_types, variant, seed=48):
        assert variant in VARIANTS
        super().__init__(fresh_cem, phi_dim, num_types, 'U1', seed)
        u1_hash = parameter_hash(self)
        reference = {n:p.detach().clone() for n,p in self.fusion.named_parameters()}
        rng = torch.get_rng_state().clone()
        self.fusion = GateChannels(num_types, phi_dim-4*num_types, variant, seed)
        assert torch.equal(rng, torch.get_rng_state())
        actual = dict(self.fusion.named_parameters())
        removed, added = set(reference)-set(actual), set(actual)-set(reference)
        assert removed == ({'gate.weight'} if variant == 'V1' else set())
        assert added == ({'backbone_gate.weight'} if variant == 'V2' else set())
        for n in set(reference)&set(actual):
            assert torch.equal(reference[n], actual[n]), n
        self.variant = variant
        new_s3 = expected_new_names(variant)
        self.info.update(structure=variant, description=DESCRIPTIONS[variant],
            reference_U1_initial_weights_sha256=u1_hash, retained_U1_tensors_exact=True,
            removed_U1_parameters=sum(reference[n].numel() for n in removed),
            removed_U1_parameter_names=sorted('fusion.'+n for n in removed),
            new_U1_parameters=sum(actual[n].numel() for n in added),
            new_U1_parameter_names=sorted('fusion.'+n for n in added),
            removed_S3_parameters=0, removed_S3_parameter_names=[],
            new_S3_parameters=sum(actual[n].numel() for n in new_s3),
            new_S3_parameter_names=sorted('fusion.'+n for n in new_s3),
            added_parameters=sum(p.numel() for p in self.fusion.parameters()),
            gate_condition='scalar' if variant == 'V1' else 'LN(event), LN(interaction), LN(current member h)',
            gate_scalar_from_fresh_U1_bias=variant == 'V1',
            backbone_condition_before_residual=True if variant == 'V2' else None)

    def _inject(self, module, args):
        if self._context is None:
            return None
        if self.variant != 'V2':
            return super()._inject(module, args)
        tensors = dict(args[0])
        hidden = tensors['x_cont']
        delta = self.fusion.readout(*self._context, hidden=hidden)
        assert delta.shape == hidden.shape
        tensors['x_cont'] = hidden+delta
        return (tensors,)

    def forward(self, batch):
        if self.variant != 'V2' or not self.fusion_enabled:
            return super().forward(batch)
        raw = batch['raw']
        scaled = (raw-self.base.numerical.center)*self.base.numerical.factor
        scaled = torch.where(self.base.numerical.passthrough, raw, scaled/torch.sqrt(1+(scaled/3).square()))
        self._context = self.fusion.encode(scaled, batch, self.phi_dim)
        try:
            return self.base(raw)
        finally:
            self._context = None
