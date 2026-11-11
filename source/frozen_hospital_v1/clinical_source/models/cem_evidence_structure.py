"""Five bounded evidence structures around the unchanged, untied CEM backbone."""
import math
import torch
from torch import nn
from models.tracehr_unified import independent_init
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('S1', 'S2', 'S3', 'S4', 'S5')


class FeedForward(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2*width, width))

    def forward(self, x):
        return x + self.layers(self.norm(x))


class EvidenceChannels(nn.Module):
    def __init__(self, num_types, global_dim, variant, seed, width=96, slots=32, dropout=.15):
        super().__init__()
        assert variant in VARIANTS
        self.variant, self.num_types, self.width = variant, num_types, width
        # Common tensors have identical fresh values across all five structures.
        with independent_init(seed, 'CEM/evidence/common'):
            self.type_embedding = nn.Embedding(num_types+1, width, padding_idx=num_types)
            self.feature_projection = nn.Sequential(nn.Linear(4, width), nn.GELU())
            self.global_projection = nn.Linear(global_dim, width)
            self.token_ffn = FeedForward(width, dropout)
            self.member_queries = nn.Parameter(.02*torch.randn(8, width))
            self.output_norm = nn.LayerNorm(width)
            self.output = nn.Linear(width, 256)
            self.residual_scale = nn.Parameter(torch.tensor(.1))
        if variant != 'S1':
            with independent_init(seed, 'CEM/evidence/relations'):
                self.latents = nn.Parameter(.02*torch.randn(slots, width))
                self.cross_attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
                self.cross_norm = nn.LayerNorm(width)
                self.latent_ffn = FeedForward(width, dropout)
                self.self_attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
                self.self_norm = nn.LayerNorm(width)
                self.self_ffn = FeedForward(width, dropout)
        if variant == 'S3':
            with independent_init(seed, 'CEM/evidence/labs'):
                self.lab_identity = nn.Embedding(32, width)
                self.lab_projection = nn.Sequential(nn.Linear(26, width), nn.GELU())
                self.lab_attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
                self.lab_norm = nn.LayerNorm(width)
                self.lab_ffn = FeedForward(width, dropout)
                self.extra_output = nn.Linear(width, 256)
                self.extra_scale = nn.Parameter(torch.tensor(.1))
        if variant == 'S4':
            from transformer.mambapy.mamba import Mamba, MambaConfig
            with independent_init(seed, 'CEM/evidence/time'):
                self.time_projection = nn.Linear(3, width)
                self.time_encoder = Mamba(MambaConfig(d_model=width, n_layers=2, pscan=False, use_cuda=False))
                self.time_norm = nn.LayerNorm(width)
                self.extra_output = nn.Linear(width, 256)
                self.extra_scale = nn.Parameter(torch.tensor(.1))

    def pool(self, values, mask=None):
        scores = torch.einsum('kd,bld->bkl', self.member_queries, values)/math.sqrt(self.width)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None], -torch.inf)
        return torch.einsum('bkl,bld->bkd', scores.softmax(-1), values)

    def forward(self, scaled, batch, phi_dim):
        ids, mask = batch['event_ids'], batch['event_mask']
        safe_ids = ids.clamp_max(self.num_types-1)
        values = torch.stack([scaled[:, i*self.num_types:(i+1)*self.num_types].gather(1, safe_ids)
                              for i in range(4)], -1)
        tokens = self.token_ffn(self.type_embedding(ids)+self.feature_projection(values))
        global_token = self.global_projection(scaled[:, 4*self.num_types:phi_dim])[:, None]
        tokens = torch.cat((global_token, tokens), 1)
        mask = torch.cat((torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask), 1)
        if self.variant == 'S1':
            evidence = self.pool(tokens, mask)
        else:
            latent = self.latents[None].expand(len(tokens), -1, -1)
            read, _ = self.cross_attention(latent, tokens, tokens, key_padding_mask=~mask, need_weights=True)
            latent = self.latent_ffn(self.cross_norm(latent+read))
            read, _ = self.self_attention(latent, latent, latent, need_weights=True)
            latent = self.self_ffn(self.self_norm(latent+read))
            evidence = self.pool(latent)
        delta = self.residual_scale*self.output(self.output_norm(evidence))
        if self.variant == 'S3':
            simple = scaled[:, phi_dim:phi_dim+352].reshape(-1, 32, 11)
            state = scaled[:, phi_dim+352:].reshape(-1, 32, 15)
            labs = self.lab_projection(torch.cat((simple, state), -1))+self.lab_identity.weight[None]
            # An unparameterized null slot makes all-missing lab records well-defined.
            labs = torch.cat((torch.zeros_like(labs[:, :1]), labs), 1)
            present = batch['raw'][:, phi_dim:phi_dim+352:11] > 0
            lab_mask = torch.cat((torch.ones_like(present[:, :1]), present), 1)
            read, _ = self.lab_attention(latent, labs, labs, key_padding_mask=~lab_mask, need_weights=True)
            cross = self.lab_ffn(self.lab_norm(latent+read))
            delta = delta + self.extra_scale*self.extra_output(self.pool(cross))
        elif self.variant == 'S4':
            counts = batch['time_counts']
            mass = counts.sum(-1, keepdim=True)
            mixture = (counts/mass.clamp_min(1)) @ self.type_embedding.weight[:self.num_types]
            pos = torch.arange(8, device=mass.device, dtype=mass.dtype)[None, :, None].expand(len(mass), -1, -1)/7
            context = torch.cat((mass.log1p(), (mass > 0).to(mass.dtype), pos), -1)
            temporal = self.time_norm(self.time_encoder(mixture+self.time_projection(context)))
            delta = delta + self.extra_scale*self.extra_output(self.pool(temporal))
        return delta.transpose(0, 1)


class EvidenceStructure(nn.Module):
    def __init__(self, fresh_cem, phi_dim, num_types, variant, seed=48):
        super().__init__()
        assert variant in VARIANTS and seed == 48
        assert not fresh_cem.main_residual_enabled and not fresh_cem.auxiliary_enabled
        assert not list(fresh_cem.fusion.parameters())
        initial = {n: p.detach().clone() for n, p in fresh_cem.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in fresh_cem.named_buffers()}
        reference = parameter_hash(fresh_cem)
        self.base = fresh_cem.base
        self.fusion = EvidenceChannels(num_types, phi_dim-4*num_types, variant, seed)
        self.phi_dim, self.members, self.variant = phi_dim, 8, variant
        self.main_residual_enabled, self.fusion_enabled, self.auxiliary_enabled = True, True, False
        layers = [m for m in self.base.net.modules() if type(m).__name__ == 'WeightLayer']
        assert len(layers) == 4
        target = layers[-1] if variant == 'S5' else layers[1]
        self._context = None
        self._hook = target.register_forward_pre_hook(self._inject)
        for n, p in self.named_parameters():
            if n.startswith('base.'): assert torch.equal(p, initial[n]), n
        for n, b in self.named_buffers():
            if n.startswith('base.'): assert torch.equal(b, buffers[n]), n
        self.info = dict(fresh_cem.info, structure=variant, reference_CEM_initial_weights_sha256=reference,
            reference_CEM_parameter_count=sum(p.numel() for p in initial.values()),
            retained_CEM_tensors_exact=True, retained_CEM_buffers_exact=True,
            fusion_position='before_classifier' if variant=='S5' else 'before_second_weight_layer',
            added_parameters=sum(p.numel() for p in self.fusion.parameters()),
            auxiliary_enabled=False, main_residual_enabled=True, learned_checkpoint_loaded=False,
            all_parameters_jointly_trainable=True, freeze=False, stop_gradient=False)

    def _inject(self, module, args):
        if self._context is None: return None
        tensors = dict(args[0])
        assert tensors['x_cont'].shape == self._context.shape
        tensors['x_cont'] = tensors['x_cont'] + self._context
        return (tensors,)

    def forward(self, batch):
        if not self.fusion_enabled: return self.base(batch['raw'])
        raw = batch['raw']
        scaled = (raw-self.base.numerical.center)*self.base.numerical.factor
        scaled = torch.where(self.base.numerical.passthrough, raw, scaled/torch.sqrt(1+(scaled/3).square()))
        self._context = self.fusion(scaled, batch, self.phi_dim)
        try: return self.base(raw)
        finally: self._context = None

    def forward_views(self, batch): return self(batch), None

    def path_losses(self, logits, labels): return self.base.loss(logits, labels), None

    def loss(self, logits, labels):
        main = self.base.loss(logits, labels)
        self._last_path_losses = torch.stack((main.detach(), main.detach()*0))
        return .5*main
