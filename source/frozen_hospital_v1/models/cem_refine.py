"""Five prespecified, independently tested refinements of S3's evidence paths."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.cem_evidence_structure import EvidenceChannels, EvidenceStructure
from models.cem_connection_ablation import DIRECT_PARAMETERS
from models.cem_gated_event import GatedEventChannels
from models.tracehr_unified import independent_init
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('U1', 'U2', 'U3', 'U4', 'U5')
DESCRIPTIONS = {
    'U1': 'input-dependent gate on the S3 event direct residual',
    'U2': 'observation-state attention bias on S3 without event direct residual',
    'U3': 'laboratory queries read event slots; no event direct residual',
    'U4': 'remove event-slot self-attention and its feedforward block; no direct residual',
    'U5': 'backbone-conditioned member queries; no event direct residual',
}


def expected_new_names(variant):
    return {'U1': {'gate.weight', 'gate.bias'},
            'U2': {'observation_bias.weight'}, 'U3': set(), 'U4': set(),
            'U5': {'query_condition.weight'}}[variant]


def expected_removed_names(variant, original_names):
    if variant == 'U1':
        return set()
    removed = set(DIRECT_PARAMETERS)
    if variant == 'U4':
        removed |= {n for n in original_names if n.startswith(('self_attention.', 'self_norm.', 'self_ffn.'))}
    return removed


class RefineChannels(EvidenceChannels):
    def __init__(self, num_types, global_dim, variant, seed):
        assert variant in VARIANTS[1:]
        super().__init__(num_types, global_dim, 'S3', seed)
        self.refinement = variant
        self.event_direct_enabled = False
        self.interaction_event_skip_enabled = variant != 'U3'
        del self.output_norm, self.output, self.residual_scale
        if variant == 'U2':
            # Existing four scaled fields: log count, first availability time,
            # time since last result, and time since its sample. No new input.
            with independent_init(seed, 'CEM/evidence/observation_bias'):
                self.observation_bias = nn.Linear(4, 1, bias=False)
        elif variant == 'U4':
            del self.self_attention, self.self_norm, self.self_ffn
        elif variant == 'U5':
            with independent_init(seed, 'CEM/evidence/query_condition'):
                self.query_condition = nn.Linear(256, self.width, bias=False)

    def encode(self, scaled, batch, phi_dim):
        ids, mask = batch['event_ids'], batch['event_mask']
        safe = ids.clamp_max(self.num_types-1)
        values = torch.stack([scaled[:, i*self.num_types:(i+1)*self.num_types].gather(1, safe)
                              for i in range(4)], -1)
        tokens = self.token_ffn(self.type_embedding(ids)+self.feature_projection(values))
        global_token = self.global_projection(scaled[:, 4*self.num_types:phi_dim])[:, None]
        tokens = torch.cat((global_token, tokens), 1)
        mask = torch.cat((torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask), 1)
        latent = self.latents[None].expand(len(tokens), -1, -1)
        read, _ = self.cross_attention(latent, tokens, tokens, key_padding_mask=~mask, need_weights=True)
        latent = self.latent_ffn(self.cross_norm(latent+read))
        if self.refinement != 'U4':
            read, _ = self.self_attention(latent, latent, latent, need_weights=True)
            latent = self.self_ffn(self.self_norm(latent+read))
        simple = scaled[:, phi_dim:phi_dim+352].reshape(-1, 32, 11)
        state = scaled[:, phi_dim+352:].reshape(-1, 32, 15)
        labs = self.lab_projection(torch.cat((simple, state), -1))+self.lab_identity.weight[None]
        labs = torch.cat((torch.zeros_like(labs[:, :1]), labs), 1)
        present = batch['raw'][:, phi_dim:phi_dim+352:11] > 0
        lab_mask = torch.cat((torch.ones_like(present[:, :1]), present), 1)
        if self.refinement == 'U3':
            # All event slots are valid. Only observed lab queries and the null
            # query participate in the final member readout.
            read, _ = self.lab_attention(labs, latent, latent, need_weights=True)
            return self.lab_ffn(self.lab_norm(labs+read)), lab_mask
        if self.refinement == 'U2':
            bias = self.observation_bias(simple[:, :, 1:5]).squeeze(-1)
            bias = torch.cat((torch.zeros_like(bias[:, :1]), bias), 1)
            bias = bias.masked_fill(~lab_mask, -torch.inf)
            attention_mask = bias[:, None, None, :].expand(-1, 4, latent.shape[1], -1)
            attention_mask = attention_mask.reshape(len(labs)*4, latent.shape[1], labs.shape[1])
            read, _ = self.lab_attention(latent, labs, labs, attn_mask=attention_mask, need_weights=True)
        else:
            read, _ = self.lab_attention(latent, labs, labs, key_padding_mask=~lab_mask, need_weights=True)
        return self.lab_ffn(self.lab_norm(latent+read)), None

    def readout(self, cross, mask=None, hidden=None):
        if self.refinement == 'U5':
            assert hidden is not None and hidden.shape == (8, len(cross), 256)
            normalized = F.layer_norm(hidden.transpose(0, 1), (256,))
            queries = self.member_queries[None] + self.query_condition(normalized)
            scores = torch.einsum('bkd,bld->bkl', queries, cross)/math.sqrt(self.width)
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None], -torch.inf)
            pooled = torch.einsum('bkl,bld->bkd', scores.softmax(-1), cross)
        else:
            pooled = self.pool(cross, mask)
        return (self.extra_scale*self.extra_output(pooled)).transpose(0, 1)

    def forward(self, scaled, batch, phi_dim, hidden=None):
        return self.readout(*self.encode(scaled, batch, phi_dim), hidden=hidden)


class RefineStructure(EvidenceStructure):
    def __init__(self, fresh_cem, phi_dim, num_types, variant, seed=48):
        assert variant in VARIANTS
        super().__init__(fresh_cem, phi_dim, num_types, 'S3', seed)
        reference_hash = parameter_hash(self)
        original = {n: p.detach().clone() for n, p in self.fusion.named_parameters()}
        rng = torch.get_rng_state().clone()
        self.fusion = (GatedEventChannels(num_types, phi_dim-4*num_types, seed) if variant == 'U1'
                       else RefineChannels(num_types, phi_dim-4*num_types, variant, seed))
        assert torch.equal(rng, torch.get_rng_state())
        actual = dict(self.fusion.named_parameters())
        removed, added = set(original)-set(actual), set(actual)-set(original)
        assert removed == expected_removed_names(variant, original)
        assert added == expected_new_names(variant)
        for n in set(original)&set(actual):
            assert torch.equal(original[n], actual[n]), n
        self.variant = variant
        self.info.update(structure=variant, description=DESCRIPTIONS[variant],
            reference_S3_initial_weights_sha256=reference_hash, retained_S3_tensors_exact=True,
            removed_S3_parameters=sum(original[n].numel() for n in removed),
            removed_S3_parameter_names=sorted('fusion.'+n for n in removed),
            new_S3_parameters=sum(actual[n].numel() for n in added),
            new_S3_parameter_names=sorted('fusion.'+n for n in added),
            event_direct_enabled=self.fusion.event_direct_enabled,
            interaction_event_skip_enabled=self.fusion.interaction_event_skip_enabled,
            added_parameters=sum(p.numel() for p in self.fusion.parameters()))

    def _inject(self, module, args):
        if self._context is None:
            return None
        if self.variant != 'U5':
            return super()._inject(module, args)
        tensors = dict(args[0])
        hidden = tensors['x_cont']
        # This is the current main-backbone tensor, not a separate forward or a
        # detached summary. Gradients flow through both conditioning and evidence.
        delta = self.fusion.readout(*self._context, hidden=hidden)
        assert delta.shape == hidden.shape
        tensors['x_cont'] = hidden + delta
        return (tensors,)

    def forward(self, batch):
        if self.variant != 'U5' or not self.fusion_enabled:
            return super().forward(batch)
        raw = batch['raw']
        scaled = (raw-self.base.numerical.center)*self.base.numerical.factor
        scaled = torch.where(self.base.numerical.passthrough, raw, scaled/torch.sqrt(1+(scaled/3).square()))
        self._context = self.fusion.encode(scaled, batch, self.phi_dim)
        try:
            return self.base(raw)
        finally:
            self._context = None
