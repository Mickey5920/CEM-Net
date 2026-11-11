"""Unmodified author Medformer graph with the already verified clinical adapter."""
import importlib.util
from pathlib import Path
import sys
import types
import torch
from models.cem_sequence_comparators import ClinicalAdapter
from scripts.cem_medformer_common import recipe
from scripts.train_tracehr_unified import seed_all

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'work/cem_med_transformer_pilot_20261103/vendor/medformer'


def author_class():
    if 'reformer_pytorch' not in sys.modules:
        unused = types.ModuleType('reformer_pytorch')
        class UnusedReformer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError('Reformer is not part of the selected Medformer graph')
        unused.LSHSelfAttention = UnusedReformer
        sys.modules['reformer_pytorch'] = unused
    sys.path.insert(0, str(VENDOR))
    spec = importlib.util.spec_from_file_location('cem_author_medformer', VENDOR / 'models/Medformer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


class Medformer(torch.nn.Module):
    def __init__(self, num_types):
        super().__init__()
        # Preserve the pilot construction order: author backbone, then adapter.
        backbone = author_class()(types.SimpleNamespace(**recipe()['model']))
        self.adapter = ClinicalAdapter(num_types)
        self.backbone = backbone

    def forward(self, *batch):
        return self.backbone(self.adapter(*batch), None, None, None)


def make_model(seed, data):
    seed_all(seed)
    return Medformer(data.num_types)


def optimizer(model):
    r = recipe()
    return torch.optim.Adam(model.parameters(), lr=r['lr'], eps=r['epsilon'], weight_decay=r['weight_decay'])
