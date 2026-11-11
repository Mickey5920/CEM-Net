"""The three prespecified deletions from S3; no new trainable mechanism."""
import torch
from models.cem_evidence_structure import EvidenceChannels, EvidenceStructure
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('S3_noDirect', 'S3_noSkip', 'S3_noDirect_noSkip')
DIRECT_PARAMETERS = {'output_norm.weight', 'output_norm.bias', 'output.weight',
                     'output.bias', 'residual_scale'}


class ConnectionChannels(EvidenceChannels):
    def __init__(self, num_types, global_dim, variant, seed):
        assert variant in VARIANTS
        super().__init__(num_types, global_dim, 'S3', seed)
        self.event_direct_enabled = variant == 'S3_noSkip'
        self.interaction_event_skip_enabled = variant == 'S3_noDirect'
        if not self.event_direct_enabled:
            del self.output_norm, self.output, self.residual_scale

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
        direct = (self.residual_scale*self.output(self.output_norm(self.pool(latent)))
                  if self.event_direct_enabled else None)
        simple = scaled[:, phi_dim:phi_dim+352].reshape(-1, 32, 11)
        state = scaled[:, phi_dim+352:].reshape(-1, 32, 15)
        labs = self.lab_projection(torch.cat((simple, state), -1))+self.lab_identity.weight[None]
        labs = torch.cat((torch.zeros_like(labs[:, :1]), labs), 1)
        present = batch['raw'][:, phi_dim:phi_dim+352:11] > 0
        lab_mask = torch.cat((torch.ones_like(present[:, :1]), present), 1)
        read, _ = self.lab_attention(latent, labs, labs, key_padding_mask=~lab_mask, need_weights=True)
        cross = self.lab_ffn(self.lab_norm(latent+read if self.interaction_event_skip_enabled else read))
        delta = self.extra_scale*self.extra_output(self.pool(cross))
        if direct is not None:
            delta = direct + delta
        return delta.transpose(0, 1)


class ConnectionStructure(EvidenceStructure):
    def __init__(self, fresh_cem, phi_dim, num_types, variant, seed=48):
        # Construct the same fresh S3, then delete only the prespecified connections.
        super().__init__(fresh_cem, phi_dim, num_types, 'S3', seed)
        full_hash = parameter_hash(self)
        reference = {n: p.detach().clone() for n, p in self.fusion.named_parameters()}
        before_rng = torch.get_rng_state().clone()
        self.fusion = ConnectionChannels(num_types, phi_dim-4*num_types, variant, seed)
        assert torch.equal(before_rng, torch.get_rng_state())
        kept = dict(self.fusion.named_parameters())
        deleted = set(reference)-set(kept)
        wanted = set() if self.fusion.event_direct_enabled else DIRECT_PARAMETERS
        assert deleted == wanted and set(kept) <= set(reference)
        for n, p in kept.items():
            assert torch.equal(p, reference[n]), n
        self.variant = variant
        self.info.update(structure=variant, reference_S3_initial_weights_sha256=full_hash,
            retained_S3_tensors_exact=True,
            removed_S3_parameter_names=sorted('fusion.'+n for n in deleted),
            removed_S3_parameters=sum(reference[n].numel() for n in deleted),
            event_direct_enabled=self.fusion.event_direct_enabled,
            interaction_event_skip_enabled=self.fusion.interaction_event_skip_enabled,
            event_query_retained=True, laboratory_attention_retained=True,
            added_parameters=sum(p.numel() for p in self.fusion.parameters()))
