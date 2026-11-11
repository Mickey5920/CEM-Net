"""Delete one output pathway from the fixed scalar-gated evidence branch."""
import hashlib
import torch
from models.cem_gate_location import GateStructure, GateChannels
from models.tracehr_unified_next import parameter_hash

DIRECT_MODULES = ('output_norm', 'output', 'residual_scale', 'gate')
INTERACTION_MODULES = ('lab_projection', 'lab_identity', 'lab_attention', 'lab_norm',
                       'lab_ffn', 'extra_output', 'extra_scale')


def removed_names(names, variant):
    prefixes = DIRECT_MODULES if variant == 'V1_noDirect' else INTERACTION_MODULES if variant == 'V1_noInteraction' else ()
    return {n for n in names if any(n == p or n.startswith(p+'.') for p in prefixes)}


class PathChannels(GateChannels):
    def forward(self, scaled, batch, phi_dim, hidden=None):
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
        if self.path_variant == 'V1_noInteraction':
            event = self.pool(latent)
            direct = self.residual_scale*self.output(self.output_norm(event))
            # Scalar gate has no input dependence in the fixed complete model.
            return (self.gate.bias.sigmoid()*direct).transpose(0, 1)
        assert self.path_variant == 'V1_noDirect'
        simple = scaled[:, phi_dim:phi_dim+352].reshape(-1, 32, 11)
        state = scaled[:, phi_dim+352:].reshape(-1, 32, 15)
        labs = self.lab_projection(torch.cat((simple, state), -1))+self.lab_identity.weight[None]
        labs = torch.cat((torch.zeros_like(labs[:, :1]), labs), 1)
        present = batch['raw'][:, phi_dim:phi_dim+352:11] > 0
        lab_mask = torch.cat((torch.ones_like(present[:, :1]), present), 1)
        read, _ = self.lab_attention(latent, labs, labs, key_padding_mask=~lab_mask, need_weights=True)
        cross = self.lab_ffn(self.lab_norm(latent+read))
        return (self.extra_scale*self.extra_output(self.pool(cross))).transpose(0, 1)


def delete_path(fusion, variant):
    assert variant in ('V1_noDirect', 'V1_noInteraction')
    fusion.__class__ = PathChannels
    fusion.path_variant = variant
    for name in DIRECT_MODULES if variant == 'V1_noDirect' else INTERACTION_MODULES:
        delattr(fusion, name)
    fusion.event_direct_enabled = variant != 'V1_noDirect'
    fusion.interaction_event_skip_enabled = variant != 'V1_noInteraction'
    return fusion


class PathStructure(GateStructure):
    def __init__(self, fresh_base, phi_dim, num_types, variant, seed):
        assert variant in ('V1_group_rate', 'V1_noDirect', 'V1_noInteraction')
        super().__init__(fresh_base, phi_dim, num_types, 'V1', seed)
        full_hash = parameter_hash(self)
        reference = {n: p.detach().clone() for n, p in self.named_parameters()}
        rng = torch.get_rng_state().clone()
        deleted = {'fusion.'+n for n in removed_names(dict(self.fusion.named_parameters()), variant)}
        if variant != 'V1_group_rate':
            delete_path(self.fusion, variant)
        actual = dict(self.named_parameters())
        assert set(reference)-set(actual) == deleted and not set(actual)-set(reference)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(p, reference[n]) for n, p in actual.items())
        self.info.update(reference_full_initial_weights_sha256=full_hash,
            full_initial_tensor_hashes={n: hashlib.sha256(p.cpu().contiguous().numpy().tobytes()).hexdigest()
                                       for n, p in reference.items()},
            deleted_parameter_names=sorted(deleted), deleted_parameters=sum(reference[n].numel() for n in deleted),
            retained_full_initial_tensors_exact=True, deletion_preserves_rng=True,
            path_variant=variant, event_direct_enabled=variant != 'V1_noDirect',
            interaction_event_skip_enabled=variant != 'V1_noInteraction',
            laboratory_interaction_enabled=variant != 'V1_noInteraction',
            added_parameters=sum(p.numel() for p in self.fusion.parameters()))
