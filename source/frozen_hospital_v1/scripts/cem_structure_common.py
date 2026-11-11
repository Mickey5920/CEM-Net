"""One authorized MIMIC screen: five structures, two endpoints, existing seed48."""
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scripts.tracehr_unified_common import Data as RawData
from scripts.tracehr_fte_binding_common import model_for as cem_model, protocol as cem_protocol, VARIANT as CEM
from scripts.tracehr_structure_common import stopping
from scripts.tracehr_structure_runtime import Optimizer, diagnostics
from scripts.tracehr_multiview_common import atomic, read, sha
from models.cem_evidence_structure import EvidenceStructure, VARIANTS
from models.tracehr_unified_next import parameter_hash

RUN = 'tracehr_cem_structure_20261023'
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (48,)
PROTOCOL_FIELDS = {'variant','structure','ablated_component','description','ablation_implementation',
    'evidence_width','evidence_slots','evidence_heads','evidence_dropout','additional_channels',
    'evidence_mixer','fusion_position','temporal_source','time_encoder'}

def expected(): return read(ROOT/'work'/RUN/'expected.json')

def matrix(database):
    assert database == 'mimic'
    return tuple((v, e, 48) for v in VARIANTS for e in (0, 1))

def protocol(variant, database, seed):
    assert (variant, 0, seed) in matrix(database)
    spec = cem_protocol(CEM, database, seed)
    spec.update(variant=variant, structure=variant, ablated_component=None,
        description='fixed CEM plus evidence structure '+variant, ablation_implementation=None,
        evidence_width=96, evidence_slots=32, evidence_heads=4, evidence_dropout=.15,
        additional_channels=2 if variant in ('S3','S4') else 1,
        evidence_mixer='set' if variant=='S1' else 'latent_relations',
        fusion_position='before_classifier' if variant=='S5' else 'before_second_weight_layer',
        temporal_source='existing availability-filtered categorical events; eight 3h bins' if variant=='S4' else None,
        time_encoder='existing Mamba; two layers; sequential scan' if variant=='S4' else None)
    return spec

class Data(RawData):
    def __init__(self, project, database, preparation, need_time=False):
        super().__init__(project, database, preparation)
        self.num_types = self.record['event_num_types']
        self.lengths = {s:np.load(preparation/(s+'_event_lengths.npy'), mmap_mode='r') for s in ('train','dev')}
        self.temporal = {}
        if need_time:
            for s in ('train','dev'):
                f=preparation/(s+'_bin_counts.npz')
                assert sha(f)==self.record['structure_artifacts'][f.name]['sha256']
                self.temporal[s]=sparse.load_npz(f)

    def batch(self, split, indices, device='cuda'):
        assert split in ('train','dev')
        raw=np.array(self.raw[split][indices], copy=True)
        present=raw[:, self.num_types:2*self.num_types]>0
        lengths=present.sum(1); width=max(int(lengths.max(initial=0)),1)
        ids=np.full((len(indices),width),self.num_types,np.int64)
        mask=np.zeros((len(indices),width),bool)
        for i,row in enumerate(present):
            active=np.flatnonzero(row); ids[i,:len(active)]=active; mask[i,:len(active)]=True
        batch=dict(raw=torch.from_numpy(raw).to(device),event_ids=torch.from_numpy(ids).to(device),
                   event_mask=torch.from_numpy(mask).to(device))
        if self.temporal:
            dense=self.temporal[split][indices].toarray().astype(np.float32).reshape(len(indices),8,self.num_types)
            batch['time_counts']=torch.from_numpy(dense).to(device)
        y=torch.from_numpy(self.targets[split]['y'][indices].astype(np.float32)).to(device)
        return batch,y

def model_for(data, preparation, variant, endpoint, seed):
    fresh=cem_model(data, preparation, CEM, endpoint, seed)
    return EvidenceStructure(fresh, data.record['phi_dim'], data.num_types, variant, seed)
