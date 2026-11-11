"""E-round: centered shared-member input and complete native random initialization."""
import copy
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.tracehr_backbone_refine import TypedNative
from models.tracehr_unified_next import fixed_to_buffers


class CenteredNative(TypedNative):
    def __init__(self, raw, y, prep, config, seed=42, centered=True, init_rows=1024):
        super().__init__(raw, y, prep, config, 8, seed, init_rows)
        self.centered = centered
        ix = np.random.RandomState(seed).permutation(len(raw))[:min(init_rows, len(raw))]
        with torch.no_grad():
            total = torch.zeros(len(self.feature_map), dtype=torch.float64)
            for indices in np.array_split(ix, max(1, math.ceil(len(ix)/64))):
                total += self.numerical(torch.as_tensor(np.array(raw[indices], dtype=np.float32))).double().sum(0)
        self.register_buffer('encoded_center', (total/len(ix)).float())
        self.info.update(centered_modulation=centered, center_fit_split='train',
                         center_updates='fixed statistic at random initialization; all learned parameters train jointly')

    def forward(self, raw):
        z = self.numerical(raw)[None]
        r = self.input_members.index_select(1, self.feature_map)[:, None]
        x = self.encoded_center+r*(z-self.encoded_center) if self.centered else r*z
        tensors = dict(x_cont=x, x_cat=torch.empty((8, len(raw), 0), dtype=torch.long, device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0, 1)


def native_parameter_hash(net):
    h = hashlib.sha256()
    for name, p in net.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FullNative(nn.Module):
    """Capture the original sklearn entry AFTER random initialization, BEFORE training.

    The private sentinel prevents even one original optimizer update. No saved
    learned state is read. Native Variable metadata is retained for its optimizer.
    """
    def __init__(self, raw, targets, endpoint, tmp_folder, seed=42, device='cuda:0'):
        super().__init__()
        sys.modules['lightning'] = None
        from pytabkit import RealMLP_TD_Classifier
        from pytabkit.models.training.nn_creator import NNCreator
        original = NNCreator.create_model
        captured = {}

        class InitializationCaptured(Exception):
            pass

        def capture(creator, ds, idxs_list):
            net = original(creator, ds, idxs_list)
            captured.update(net=net, static=creator.static_model, hp=creator.hp_manager,
                            native_initial_trainable_sha256=native_parameter_hash(net),
                            internal_seeds=[list(map(int, x.sub_split_seeds)) for x in idxs_list])
            raise InitializationCaptured()

        estimator = RealMLP_TD_Classifier(device=device, random_state=seed, n_cv=1, n_refit=0, n_threads=4,
            tmp_folder=str(tmp_folder), verbosity=0, n_epochs=256, batch_size=256, predict_batch_size=256,
            val_metric_name='tracehr_1_ap', use_ls=False, use_early_stopping=False, num_emb_type='pbld')
        cfg = copy.deepcopy(estimator.get_config())
        NNCreator.create_model = capture
        try:
            estimator.fit(raw['train'], targets['train'][:, endpoint].astype(np.int64),
                          X_val=raw['dev'], y_val=targets['dev'][:, endpoint].astype(np.int64))
        except InitializationCaptured:
            pass
        finally:
            NNCreator.create_model = original
        assert captured and captured['internal_seeds']==[[int(np.random.RandomState(seed).randint(0, 2**31-1))]]
        self.net, self.static, self.hp = captured['net'], captured['static'], captured['hp']
        self.members = 1
        self.register_parameter('input_members', None)
        x = torch.as_tensor(np.array(raw['train'][:16], dtype=np.float32), device=device)
        self.eval()
        with torch.no_grad():
            before = self(x).clone()
        fixed = fixed_to_buffers(self.net)+['static.'+n for n in fixed_to_buffers(self.static)]
        with torch.no_grad():
            drift = float((self(x)-before).abs().max())
        assert drift==0 and native_parameter_hash(self.net)==captured['native_initial_trainable_sha256']
        self.info = dict(config=cfg, fixed_statistics=fixed, internal_seeds=captured['internal_seeds'],
                         native_initial_trainable_sha256=captured['native_initial_trainable_sha256'],
                         initial_function_wrapper_max_logit_drift=drift, original_optimizer_updates=0,
                         initialization_split='train', native_full_frontend=True, members=1,
                         shared_weight_layers=4, independent_member_heads=0, hidden_member_adapters=0,
                         adaptation='complete original native initialization; common E training sampler and microbatch',
                         loop_differences=['NumPy default_rng training permutation',
                                           'microbatch 128 with gradient accumulation to effective batch 256',
                                           'development AP checked each epoch; boundary screening stopping'])
        assert all(p.requires_grad for p in self.parameters())

    def forward(self, raw):
        tensors = dict(x_cont=raw[None], x_cat=torch.empty((1, len(raw), 0), dtype=torch.long, device=raw.device))
        return self.net(self.static(tensors))['x_cont'].transpose(0, 1)

    def loss(self, logits, y):
        return F.cross_entropy(logits[:, 0], y.long())

    def probabilities(self, raw):
        return self(raw).softmax(-1)[:, 0, 1:2]
