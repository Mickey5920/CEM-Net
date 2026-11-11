"""One input-dependent gate on S3's event direct residual; all other paths retained."""
import torch
from torch import nn
from torch.nn import functional as F
from models.cem_evidence_structure import EvidenceChannels, EvidenceStructure
from models.tracehr_unified import independent_init
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('S3_gatedDirect',)
GATE_NAMES = {'gate.weight', 'gate.bias'}


class GatedEventChannels(EvidenceChannels):
    def __init__(self, num_types, global_dim, seed):
        super().__init__(num_types, global_dim, 'S3', seed)
        with independent_init(seed, 'CEM/evidence/event_gate'):
            self.gate = nn.Linear(2*self.width, 1)
        self.event_direct_enabled = self.interaction_event_skip_enabled = True
        self.last_gate_summary = None

    def forward(self, scaled, batch, phi_dim):
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
        gate_input = torch.cat((F.layer_norm(event, (self.width,)),
                                F.layer_norm(interaction, (self.width,))), -1)
        g = self.gate(gate_input).sigmoid()
        delta = g*direct + self.extra_scale*self.extra_output(interaction)
        if not torch.is_grad_enabled():
            self.last_gate_summary = dict(scope='last_inference_batch_only',
                rows=len(g), mean=float(g.mean()), std=float(g.std(unbiased=False)),
                minimum=float(g.min()), maximum=float(g.max()))
        return delta.transpose(0, 1)


class GatedEventStructure(EvidenceStructure):
    def __init__(self, fresh_cem, phi_dim, num_types, variant, seed=48):
        assert variant in VARIANTS
        super().__init__(fresh_cem, phi_dim, num_types, 'S3', seed)
        full_hash = parameter_hash(self)
        reference = {n: p.detach().clone() for n, p in self.fusion.named_parameters()}
        before_rng = torch.get_rng_state().clone()
        self.fusion = GatedEventChannels(num_types, phi_dim-4*num_types, seed)
        assert torch.equal(before_rng, torch.get_rng_state())
        actual = dict(self.fusion.named_parameters())
        assert set(actual)-set(reference) == GATE_NAMES and set(reference) <= set(actual)
        for n, p in reference.items():
            assert torch.equal(actual[n], p), n
        assert sum(actual[n].numel() for n in GATE_NAMES) == 193
        self.variant = variant
        self.info.update(structure=variant, reference_S3_initial_weights_sha256=full_hash,
            retained_S3_tensors_exact=True, removed_S3_parameters=0, removed_S3_parameter_names=[],
            gate_added_parameters=193, gate_parameter_names=sorted('fusion.'+n for n in GATE_NAMES),
            event_direct_enabled=True, interaction_event_skip_enabled=True, event_direct_gated=True,
            gate_input='normalized event and interaction pooled representations',
            gate_initialization='independent seeded default Linear random initialization',
            event_query_retained=True, laboratory_attention_retained=True,
            added_parameters=sum(p.numel() for p in self.fusion.parameters()))
