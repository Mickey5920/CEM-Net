import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from Experiments.scripts.exp2c_metrics import evaluate_multitask, select_thresholds_on_dev
from models.amhp_encoder_adapter import AMHPEncoderAdapter
from models.clinical_2024plus_adapters import (
    ClinicalCOREBEHRTAdapter,
    ClinicalEHRMambaAdapter,
    ClinicalHyMaTEAdapter,
)
from models.clinical_gru import ClinicalGRUEncoder
from models.clinical_decoupled_mtpp_ode import ClinicalDecoupledMTPPODEEncoder
from models.clinical_hawkes_mamba import (
    ClinicalHawkesMambaEncoder,
    is_trace_clean_core_intervention,
    resolve_hera_intervention,
)
from models.trace_representation_controls import (
    TraceRepresentationControlledEncoder,
    is_trace_representation_intervention,
    resolve_trace_intervention,
)
from models.clinical_heads import (
    BinaryMultiTaskHead,
    BoundedDeltaEndpointHead,
    ClinicalPredictionModel,
    EndpointConditionedMultiTaskHead,
    masked_bce_with_logits,
    pairwise_logistic_ranking_loss,
)
from models.clinical_lstm import ClinicalLSTMEncoder
from models.clinical_mamba import ClinicalMambaEncoder
from models.clinical_retain import ClinicalRETAINEncoder
from models.clinical_rescnn import ClinicalResCNNEncoder
from models.clinical_rnn import ClinicalRNNEncoder
from models.clinical_s2p2 import ClinicalS2P2Encoder
from models.clinical_static import ClinicalStaticMLPEncoder
from models.clinical_tlstm import ClinicalTLSTMEncoder
from models.clinical_transformer import ClinicalTransformerEncoder
from models.clinical_tracehr import ClinicalTRACEHREncoder
from models.clinical_xtsformer import ClinicalXTSFormerEncoder
from models.mhp_encoder_adapter import MHPEncoderAdapter
from preprocess.ClinicalDataset import get_clinical_dataloader, infer_source_names

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train Experiment 2C clinical deep model.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--model',
        choices=[
            'gru',
            'rnn',
            'lstm',
            'mlp',
            'resmlp',
            'rescnn',
            'transformer',
            'xtsformer',
            'mamba',
            'ehrmamba',
            'core_behrt',
            'hymate',
            'hawkes_mamba',
            'tracehr',
            's2p2',
            'decoupled_mtpp_ode',
            'tlstm',
            'retain',
            'mhp',
            'amhp',
            'mhp_pretrained',
            'amhp_pretrained',
        ],
        required=True,
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--monitor', choices=['mean_primary_auprc', 'mean_primary_auroc'], default='mean_primary_auprc')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--d_inner', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument(
        '--sequence_n_layers',
        type=int,
        default=None,
        help='Sequence-backbone depth for hawkes_mamba; defaults to --n_layers.',
    )
    parser.add_argument(
        '--state_n_layers',
        type=int,
        default=None,
        help='Clinical-state MLP depth for hawkes_mamba; defaults to --n_layers.',
    )
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_k', type=int, default=16)
    parser.add_argument('--d_v', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling', choices=['last', 'mean', 'attention'], default='last')
    parser.add_argument('--time_scale_hours', type=float, default=None)
    parser.add_argument('--hawkes_context_scale', type=float, default=1.0)
    parser.add_argument('--static_context_scale', type=float, default=0.0)
    parser.add_argument('--static_feature_set', choices=['basic', 'rich'], default='basic')
    parser.add_argument('--static_time_decay_hours', type=float, default=12.0)
    parser.add_argument('--static_hidden_dim', type=int, default=None)
    parser.add_argument(
        '--trace_scale_mode',
        choices=['multiscale', 'single'],
        default='multiscale',
        help='TRACEHR accumulated-activity scales; single is the matched scale ablation.',
    )
    parser.add_argument('--trace_num_scales', type=int, default=4)
    parser.add_argument('--trace_single_scale_hours', type=float, default=12.0)
    parser.add_argument(
        '--trace_mixer_mode',
        choices=['relational', 'deepset'],
        default='relational',
        help='Latent relational set mixer or its interaction-free DeepSets ablation.',
    )
    parser.add_argument(
        '--trace_endpoint_conditioning',
        choices=['task_specific', 'shared'],
        default='task_specific',
        help='Use endpoint-specific evidence queries or one shared-query ablation.',
    )
    parser.add_argument('--trace_num_latents', type=int, default=16)
    parser.add_argument('--trace_ffn_dim', type=int, default=None)
    parser.add_argument('--trace_evidence_scale', type=float, default=1.25)
    parser.add_argument(
        '--trace_temporal_mode',
        choices=['mamba', 'none'],
        default='none',
        help=(
            'Order-sensitive temporal backbone. The CLI default preserves the v2 '
            'evidence-only run; paper-aligned queues must explicitly select mamba.'
        ),
    )
    parser.add_argument(
        '--trace_evidence_path',
        choices=['enabled', 'disabled'],
        default='enabled',
        help='Enable accumulated evidence; disabled is the trajectory-only control.',
    )
    parser.add_argument(
        '--trace_evidence_input',
        choices=['observed', 'trajectory_tiled'],
        default='observed',
        help=(
            'Observed accumulated evidence or a parameter-matched control fed only '
            'by a deterministically tiled trajectory representation.'
        ),
    )
    parser.add_argument(
        '--trace_evidence_feature_mask',
        choices=[
            'none',
            'burden_coverage',
            'recency_activity',
            'care_source',
            'observation_intensity',
        ],
        default='none',
        help='Remove one accumulated-evidence family while retaining model capacity.',
    )
    parser.add_argument(
        '--trace_time_gap_mode',
        choices=['observed', 'zero'],
        default='observed',
        help='Use validated event gaps or zero only the explicit gap input.',
    )
    parser.add_argument(
        '--trace_upgrade_variant',
        choices=[
            'current',
            'u1_guarded',
            'u2_anchored',
            'u3_hierarchical',
            'u4_delta',
        ],
        default='current',
        help='Complete TRACEHR upgrade architecture; current preserves the prior path.',
    )
    parser.add_argument('--trace_anchor_static_scale', type=float, default=1.25)
    parser.add_argument('--trace_anchor_static_hidden_dim', type=int, default=None)
    parser.add_argument('--trace_delta_logit_bound', type=float, default=0.5)
    parser.add_argument(
        '--trace_member',
        choices=['A', 'B', 'single'],
        default='single',
        help='Prespecified TRACEHR member label recorded in run metadata.',
    )
    parser.add_argument(
        '--fusion_mode',
        choices=['fixed_add', 'normalized_residual', 'adaptive_gate'],
        default='fixed_add',
        help='Trajectory-state fusion used by hawkes_mamba when the state branch is enabled.',
    )
    parser.add_argument(
        '--trajectory_residual_init',
        type=float,
        default=0.25,
        help='Initial trajectory residual weight for learned fusion modes.',
    )
    parser.add_argument('--hawkes_initial_decay_hours', type=float, default=None)
    parser.add_argument(
        '--hera_intervention',
        choices=[
            'full',
            'wo_hawkes',
            'wo_rich_static',
            'wo_ranking',
            'transformer_backbone',
            'clean_core',
            'clean_core_trajectory_only',
            'clean_core_state_only',
            'clean_core_transformer',
            'clean_core_basic_state',
            'state_only',
            'trajectory_reversed_order',
            'trajectory_no_time',
            'wo_state_burden',
            'wo_state_recency',
            'wo_state_source',
            'wo_state_intensity',
        ],
        default=None,
        help='HERA/TRACE configuration; omitted means the historical Full path.',
    )
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None)
    parser.add_argument('--auto_find_pretrained', action='store_true')
    parser.add_argument('--primary_tasks', nargs='+', default=None)
    parser.add_argument('--ranking_loss_weight', type=float, default=0.0)
    parser.add_argument('--ranking_tasks', nargs='+', default=None)
    parser.add_argument('--ranking_max_pairs_per_task', type=int, default=4096)
    parser.add_argument('--save_diagnostic_stats', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_eval_samples', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument(
        '--evaluation_scope',
        choices=['dev_and_test', 'dev_only'],
        default='dev_and_test',
        help='dev_only prevents loading or evaluating the held-out test split during development.',
    )
    parser.add_argument(
        '--baseline_stage',
        '--baseline-stage',
        dest='baseline_stage',
        choices=[
            'static_reproduction',
            'paperfaithful_v1',
            'tuned_baseline_v1',
            'main_method_v1',
            'main_method_v2',
        ],
        default='static_reproduction',
        help='Artifact label. main_method_v1 is for rank-aware Hawkes-Mamba runs, not paper-faithful baselines.',
    )
    parser.add_argument(
        '--progress',
        choices=['log', 'tqdm', 'both', 'none'],
        default='log',
        help='Training progress display: structured logs, tqdm bars, both, or none.',
    )
    return parser.parse_args()


def log_message(level, message, blank_before=False, blank_after=False):
    if blank_before:
        print('', flush=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {message}', flush=True)
    if blank_after:
        print('', flush=True)


def log_stage(name):
    print('', flush=True)
    print('=' * 88, flush=True)
    log_message('STAGE', name)
    print('=' * 88, flush=True)


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def label_prevalence(samples, tasks):
    output = {}
    for task in tasks:
        values = [sample.get('labels', {}).get(task) for sample in samples]
        values = [value for value in values if value is not None]
        output[task] = float(np.mean(values)) if values else None
    return output


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_pickle(path):
    import pickle

    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin-1')


def load_splits(data_dir, evaluation_scope='dev_and_test'):
    data_dir = Path(data_dir)
    split_names = ['train', 'dev']
    if evaluation_scope == 'dev_and_test':
        split_names.append('test')
    elif evaluation_scope != 'dev_only':
        raise ValueError(f'unknown evaluation_scope={evaluation_scope!r}')
    payloads = {split: load_pickle(data_dir / f'{split}.pkl') for split in split_names}
    tasks = payloads['train']['tasks']
    num_types = int(payloads['train']['dim_process'])
    samples = {split: payloads[split][split] for split in split_names}
    return samples, tasks, num_types


def limit_samples_for_smoke(samples, max_train_samples=None, max_eval_samples=None):
    limits = {'train': max_train_samples, 'dev': max_eval_samples, 'test': max_eval_samples}
    return {
        split: values
        if limits.get(split) is None or limits[split] < 0
        else values[: limits[split]]
        for split, values in samples.items()
    }


def resolve_time_scale_hours(data_dir, requested):
    if requested is not None:
        return float(requested)
    metadata_path = Path(data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        if metadata.get('window_hours') is not None:
            return float(metadata['window_hours'])
    return 24.0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    if requested.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu'), 'CUDA unavailable; using CPU'
    device = torch.device(requested)
    if device.type == 'cuda' and device.index is not None and device.index >= torch.cuda.device_count():
        return torch.device('cpu'), f'{requested} unavailable; using CPU'
    return device, None


def compute_pos_weight(samples, tasks, device):
    values = []
    for task in tasks:
        labels = np.asarray([sample['labels'][task] for sample in samples], dtype=float)
        positives = float(labels.sum())
        negatives = float(len(labels) - positives)
        values.append(1.0 if positives <= 0 else negatives / positives)
    return torch.tensor(values, dtype=torch.float32, device=device)


def resolve_task_indices(tasks, selected_tasks):
    selected_tasks = selected_tasks or tasks
    index_by_task = {task: index for index, task in enumerate(tasks)}
    missing = [task for task in selected_tasks if task not in index_by_task]
    if missing:
        raise ValueError(f'Unknown ranking task(s): {missing}; available tasks={tasks}')
    return [index_by_task[task] for task in selected_tasks], list(selected_tasks)


def validate_stage_objective(args):
    if args.ranking_loss_weight < 0:
        raise ValueError('--ranking_loss_weight must be non-negative')
    if args.static_context_scale < 0:
        raise ValueError('--static_context_scale must be non-negative')
    if args.static_time_decay_hours <= 0:
        raise ValueError('--static_time_decay_hours must be positive')
    if args.static_hidden_dim is not None and args.static_hidden_dim <= 0:
        raise ValueError('--static_hidden_dim must be positive when provided')
    if getattr(args, 'sequence_n_layers', None) is not None and args.sequence_n_layers <= 0:
        raise ValueError('--sequence_n_layers must be positive when provided')
    if getattr(args, 'state_n_layers', None) is not None and args.state_n_layers <= 0:
        raise ValueError('--state_n_layers must be positive when provided')
    if int(getattr(args, 'trace_num_scales', 4)) <= 0:
        raise ValueError('--trace_num_scales must be positive')
    if float(getattr(args, 'trace_single_scale_hours', 12.0)) <= 0:
        raise ValueError('--trace_single_scale_hours must be positive')
    if int(getattr(args, 'trace_num_latents', 16)) <= 0:
        raise ValueError('--trace_num_latents must be positive')
    if getattr(args, 'trace_ffn_dim', None) is not None and args.trace_ffn_dim <= 0:
        raise ValueError('--trace_ffn_dim must be positive when provided')
    if float(getattr(args, 'trace_evidence_scale', 1.25)) <= 0:
        raise ValueError('--trace_evidence_scale must be positive')
    if float(getattr(args, 'trace_anchor_static_scale', 1.25)) <= 0:
        raise ValueError('--trace_anchor_static_scale must be positive')
    if (
        getattr(args, 'trace_anchor_static_hidden_dim', None) is not None
        and args.trace_anchor_static_hidden_dim <= 0
    ):
        raise ValueError('--trace_anchor_static_hidden_dim must be positive when provided')
    if float(getattr(args, 'trace_delta_logit_bound', 0.5)) <= 0:
        raise ValueError('--trace_delta_logit_bound must be positive')
    if args.model == 'tracehr' and int(args.d_model) % int(args.n_head) != 0:
        raise ValueError('TRACEHR requires --d_model divisible by --n_head')
    if (
        args.model == 'tracehr'
        and getattr(args, 'trace_temporal_mode', 'mamba') == 'none'
        and getattr(args, 'trace_evidence_path', 'enabled') == 'disabled'
    ):
        raise ValueError('TRACEHR requires at least one enabled pathway')
    if (
        args.model == 'tracehr'
        and getattr(args, 'trace_evidence_input', 'observed') == 'trajectory_tiled'
        and (
            getattr(args, 'trace_temporal_mode', 'mamba') == 'none'
            or getattr(args, 'trace_evidence_path', 'enabled') == 'disabled'
        )
    ):
        raise ValueError('trajectory_tiled evidence requires both TRACEHR pathways')
    if not 0.0 < float(getattr(args, 'trajectory_residual_init', 0.25)) < 1.0:
        raise ValueError('--trajectory_residual_init must be strictly between 0 and 1')
    fusion_mode = getattr(args, 'fusion_mode', 'fixed_add')
    if args.model != 'hawkes_mamba' and fusion_mode != 'fixed_add':
        raise ValueError('--fusion_mode is only configurable with --model hawkes_mamba')
    if args.ranking_loss_weight > 0 and args.baseline_stage == 'paperfaithful_v1':
        raise ValueError('rank-aware loss cannot be labeled paperfaithful_v1; use --baseline_stage main_method_v1')
    validate_clean_core_variant_configuration(args)


def validate_clean_core_variant_configuration(args):
    variant = getattr(args, 'hera_intervention', None)
    if variant == 'clean_core_trajectory_only' and float(args.static_context_scale) != 0.0:
        raise ValueError('clean_core_trajectory_only requires --static_context_scale 0 for true state removal')
    if variant == 'clean_core_state_only' and float(args.static_context_scale) <= 0.0:
        raise ValueError('clean_core_state_only requires --static_context_scale > 0')
    if variant == 'clean_core_basic_state' and args.static_feature_set != 'basic':
        raise ValueError('clean_core_basic_state requires --static_feature_set basic')
    if variant == 'state_only' and float(args.static_context_scale) <= 0.0:
        raise ValueError('state_only requires --static_context_scale > 0')
    if variant in {
        'wo_state_burden',
        'wo_state_recency',
        'wo_state_source',
        'wo_state_intensity',
    }:
        if float(args.static_context_scale) <= 0.0 or args.static_feature_set != 'rich':
            raise ValueError(f'{variant} requires an enabled rich state branch')


def resolve_hera_training_contract(args):
    requested = getattr(args, 'hera_intervention', None)
    if args.model != 'hawkes_mamba':
        if requested is not None:
            raise ValueError('--hera_intervention is only valid with --model hawkes_mamba')
        return resolve_hera_intervention('full')
    requested = requested or 'full'
    if is_trace_representation_intervention(requested):
        return resolve_trace_intervention(requested)
    return resolve_hera_intervention(requested)


def is_trace_clean_core(args):
    return args.model == 'hawkes_mamba' and is_trace_clean_core_intervention(
        getattr(args, 'hera_intervention', None)
    )


def combine_hera_losses(bce_loss, ranking_loss, ranking_loss_weight, ranking_objective_gate):
    return bce_loss + float(ranking_objective_gate) * float(ranking_loss_weight) * ranking_loss


def intervention_audit_from_args(args):
    if args.model == 'tracehr':
        return {
            'schema_version': 2,
            'variant': 'tracehr',
            'trace_scale_mode': args.trace_scale_mode,
            'trace_num_scales': int(args.trace_num_scales),
            'trace_single_scale_hours': float(args.trace_single_scale_hours),
            'trace_mixer_mode': args.trace_mixer_mode,
            'trace_endpoint_conditioning': args.trace_endpoint_conditioning,
            'trace_num_latents': int(args.trace_num_latents),
            'trace_ffn_dim': int(args.trace_ffn_dim or 2 * args.d_model),
            'trace_evidence_scale': float(args.trace_evidence_scale),
            'trace_temporal_mode': args.trace_temporal_mode,
            'trace_evidence_path': args.trace_evidence_path,
            'trace_evidence_input': args.trace_evidence_input,
            'trace_evidence_feature_mask': args.trace_evidence_feature_mask,
            'trace_time_gap_mode': args.trace_time_gap_mode,
            'trace_upgrade_variant': args.trace_upgrade_variant,
            'trace_anchor_static_scale': float(args.trace_anchor_static_scale),
            'trace_anchor_static_hidden_dim': int(
                args.trace_anchor_static_hidden_dim or args.d_model
            ),
            'trace_delta_logit_bound': float(args.trace_delta_logit_bound),
            'trace_member': args.trace_member,
            'trajectory_enabled': args.trace_temporal_mode != 'none',
            'evidence_enabled': args.trace_evidence_path == 'enabled',
            'ranking_enabled': bool(args.ranking_loss_weight > 0),
        }
    contract = resolve_hera_training_contract(args)
    full = resolve_hera_intervention('full')
    changed_fields = sorted(
        key
        for key, value in contract.items()
        if value != full.get(key, 1.0 if key == 'trajectory_fusion_gate' else None)
    )
    trajectory_fusion_gate = float(contract.get('trajectory_fusion_gate', 1.0))
    return {
        'schema_version': 1,
        'variant': getattr(args, 'hera_intervention', None) or 'full',
        'changed_fields': changed_fields,
        **contract,
        'trajectory_fusion_gate': trajectory_fusion_gate,
        'static_feature_set': getattr(args, 'static_feature_set', None),
        'trajectory_enabled': bool(trajectory_fusion_gate != 0.0),
        'nominal_hawkes_context_scale': float(args.hawkes_context_scale),
        'effective_hawkes_context_scale': (
            float(args.hawkes_context_scale) * float(contract['hawkes_fusion_gate'])
        ),
        'nominal_static_context_scale': float(args.static_context_scale),
        'effective_static_context_scale': (
            float(args.static_context_scale) * float(contract['rich_static_fusion_gate'])
        ),
        'nominal_ranking_loss_weight': float(args.ranking_loss_weight),
        'effective_ranking_loss_weight': (
            float(args.ranking_loss_weight) * float(contract['ranking_objective_gate'])
        ),
        'decayed_history_enabled': not is_trace_clean_core(args),
        'sequence_n_layers': int(
            getattr(args, 'sequence_n_layers', None) or getattr(args, 'n_layers', 2)
        ),
        'state_n_layers': int(
            getattr(args, 'state_n_layers', None) or getattr(args, 'n_layers', 2)
        ),
        'fusion_mode': getattr(args, 'fusion_mode', 'fixed_add'),
        'trajectory_residual_init': float(getattr(args, 'trajectory_residual_init', 0.25)),
    }


def training_objective_description(args, ranking_tasks):
    base = 'multi-task weighted BCE on post-landmark clinical labels'
    contract = resolve_hera_training_contract(args)
    effective_weight = float(args.ranking_loss_weight) * float(contract['ranking_objective_gate'])
    if effective_weight <= 0:
        return base
    if args.model != 'hawkes_mamba':
        return (
            f'{base} + alpha={args.ranking_loss_weight:g} pairwise logistic ranking '
            f'on tasks={ranking_tasks} with max_pairs_per_task={args.ranking_max_pairs_per_task}'
        )
    return (
        f'{base} + gate={contract["ranking_objective_gate"]:g} '
        f'x alpha={args.ranking_loss_weight:g} pairwise logistic ranking '
        f'on tasks={ranking_tasks} with max_pairs_per_task={args.ranking_max_pairs_per_task}'
    )


def model_config_from_args(args):
    config = {
        'model': args.model,
        'd_model': int(args.d_model),
        'hidden_dim': int(args.hidden_dim),
        'd_inner': int(args.d_inner),
        'n_layers': int(args.n_layers),
        'n_head': int(args.n_head),
        'd_k': int(args.d_k),
        'd_v': int(args.d_v),
        'dropout': float(args.dropout),
        'pooling': args.pooling,
        'time_scale_hours': None if args.time_scale_hours is None else float(args.time_scale_hours),
        'hawkes_context_scale': float(args.hawkes_context_scale),
        'static_context_scale': float(args.static_context_scale),
        'static_feature_set': args.static_feature_set,
        'static_time_decay_hours': float(args.static_time_decay_hours),
        'static_hidden_dim': None if args.static_hidden_dim is None else int(args.static_hidden_dim),
        'hawkes_initial_decay_hours': (
            None if args.hawkes_initial_decay_hours is None else float(args.hawkes_initial_decay_hours)
        ),
        'static_source_dim': int(getattr(args, 'static_source_dim', 0)),
        'static_source_names': list(getattr(args, 'static_source_names', [])),
    }
    if args.model == 'tracehr':
        config.update(
            {
                'trace_scale_mode': getattr(args, 'trace_scale_mode', 'multiscale'),
                'trace_num_scales': int(getattr(args, 'trace_num_scales', 4)),
                'trace_single_scale_hours': float(
                    getattr(args, 'trace_single_scale_hours', 12.0)
                ),
                'trace_mixer_mode': getattr(args, 'trace_mixer_mode', 'relational'),
                'trace_endpoint_conditioning': getattr(
                    args, 'trace_endpoint_conditioning', 'task_specific'
                ),
                'trace_num_latents': int(getattr(args, 'trace_num_latents', 16)),
                'trace_ffn_dim': int(
                    getattr(args, 'trace_ffn_dim', None) or 2 * args.d_model
                ),
                'trace_evidence_scale': float(
                    getattr(args, 'trace_evidence_scale', 1.25)
                ),
                'trace_temporal_mode': getattr(args, 'trace_temporal_mode', 'mamba'),
                'trace_evidence_path': getattr(args, 'trace_evidence_path', 'enabled'),
                'trace_evidence_input': getattr(args, 'trace_evidence_input', 'observed'),
                'trace_evidence_feature_mask': getattr(
                    args, 'trace_evidence_feature_mask', 'none'
                ),
                'trace_time_gap_mode': getattr(args, 'trace_time_gap_mode', 'observed'),
                'trace_upgrade_variant': getattr(args, 'trace_upgrade_variant', 'current'),
                'trace_anchor_static_scale': float(
                    getattr(args, 'trace_anchor_static_scale', 1.25)
                ),
                'trace_anchor_static_hidden_dim': int(
                    getattr(args, 'trace_anchor_static_hidden_dim', None) or args.d_model
                ),
                'trace_delta_logit_bound': float(
                    getattr(args, 'trace_delta_logit_bound', 0.5)
                ),
                'trace_member': getattr(args, 'trace_member', 'single'),
            }
        )
    if args.model == 'hawkes_mamba':
        config['hera_intervention'] = getattr(args, 'hera_intervention', None) or 'full'
        config['decayed_history_enabled'] = not is_trace_clean_core(args)
        config['sequence_n_layers'] = int(
            getattr(args, 'sequence_n_layers', None) or args.n_layers
        )
        config['state_n_layers'] = int(getattr(args, 'state_n_layers', None) or args.n_layers)
        config['fusion_mode'] = getattr(args, 'fusion_mode', 'fixed_add')
        config['trajectory_residual_init'] = float(
            getattr(args, 'trajectory_residual_init', 0.25)
        )
        config.update(resolve_hera_training_contract(args))
        config['trajectory_fusion_gate'] = float(
            resolve_hera_training_contract(args).get('trajectory_fusion_gate', 1.0)
        )
    return config


def determinism_config_from_args(args):
    return {
        'seed': int(args.seed),
        'python_random_seeded': True,
        'numpy_seeded': True,
        'torch_seeded': True,
        'torch_cuda_seeded': bool(torch.cuda.is_available()),
        'cudnn_deterministic': bool(getattr(torch.backends.cudnn, 'deterministic', False)),
        'cudnn_benchmark': bool(getattr(torch.backends.cudnn, 'benchmark', False)),
        'dataloader_seeded': True,
    }


def build_model(args, num_types, num_tasks):
    if args.model == 'gru':
        encoder = ClinicalGRUEncoder(
            num_types=num_types,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.hidden_dim
    elif args.model == 'rnn':
        encoder = ClinicalRNNEncoder(
            num_types=num_types,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.hidden_dim
    elif args.model == 'lstm':
        encoder = ClinicalLSTMEncoder(
            num_types=num_types,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.hidden_dim
    elif args.model in {'mlp', 'resmlp'}:
        encoder = ClinicalStaticMLPEncoder(
            num_types=num_types,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            residual=args.model == 'resmlp',
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'rescnn':
        encoder = ClinicalResCNNEncoder(
            num_types=num_types,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'transformer':
        encoder = ClinicalTransformerEncoder(
            num_types=num_types,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'xtsformer':
        encoder = ClinicalXTSFormerEncoder(
            num_types=num_types,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'mamba':
        encoder = ClinicalMambaEncoder(
            num_types=num_types,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'ehrmamba':
        encoder = ClinicalEHRMambaAdapter(
            num_types=num_types,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'core_behrt':
        encoder = ClinicalCOREBEHRTAdapter(
            num_types=num_types,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'hymate':
        encoder = ClinicalHyMaTEAdapter(
            num_types=num_types,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'tracehr':
        encoder = ClinicalTRACEHREncoder(
            num_types=num_types,
            num_tasks=num_tasks,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
            source_dim=getattr(args, 'static_source_dim', 0),
            num_scales=args.trace_num_scales,
            scale_mode=args.trace_scale_mode,
            single_scale_hours=args.trace_single_scale_hours,
            mixer_mode=args.trace_mixer_mode,
            endpoint_conditioning=args.trace_endpoint_conditioning,
            num_latents=args.trace_num_latents,
            n_head=args.n_head,
            ffn_dim=args.trace_ffn_dim,
            evidence_scale=args.trace_evidence_scale,
            temporal_mode=args.trace_temporal_mode,
            evidence_enabled=args.trace_evidence_path == 'enabled',
            evidence_input_mode=args.trace_evidence_input,
            evidence_feature_group_mask=args.trace_evidence_feature_mask,
            time_gap_mode=args.trace_time_gap_mode,
            upgrade_variant=args.trace_upgrade_variant,
            anchor_static_scale=args.trace_anchor_static_scale,
            anchor_static_hidden_dim=args.trace_anchor_static_hidden_dim,
            hawkes_context_scale=args.hawkes_context_scale,
            initial_decay_hours=args.hawkes_initial_decay_hours,
            sequence_n_layers=getattr(args, 'sequence_n_layers', None),
        )
        head_dim = args.d_model
    elif args.model == 's2p2':
        encoder = ClinicalS2P2Encoder(
            num_types=num_types,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model == 'decoupled_mtpp_ode':
        encoder = ClinicalDecoupledMTPPODEEncoder(
            num_types=num_types,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.hidden_dim
    elif args.model == 'hawkes_mamba':
        intervention = resolve_hera_training_contract(args)
        encoder_class = (
            TraceRepresentationControlledEncoder
            if is_trace_representation_intervention(getattr(args, 'hera_intervention', None))
            else ClinicalHawkesMambaEncoder
        )
        encoder_kwargs = {
            'num_types': num_types,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'pooling': args.pooling,
            'time_scale_hours': args.time_scale_hours,
            'hawkes_context_scale': args.hawkes_context_scale,
            'static_context_scale': args.static_context_scale,
            'static_feature_set': args.static_feature_set,
            'static_source_dim': getattr(args, 'static_source_dim', 0),
            'static_time_decay_hours': args.static_time_decay_hours,
            'static_hidden_dim': args.static_hidden_dim,
            'initial_decay_hours': args.hawkes_initial_decay_hours,
            'sequence_backbone': intervention['sequence_backbone'],
            'trajectory_fusion_gate': intervention.get('trajectory_fusion_gate', 1.0),
            'hawkes_fusion_gate': intervention['hawkes_fusion_gate'],
            'rich_static_fusion_gate': intervention['rich_static_fusion_gate'],
            'decayed_history_enabled': not is_trace_clean_core(args),
            'transformer_n_head': args.n_head,
            'transformer_ffn_dim': intervention['transformer_ffn_dim'],
            'sequence_n_layers': getattr(args, 'sequence_n_layers', None),
            'state_n_layers': getattr(args, 'state_n_layers', None),
            'fusion_mode': getattr(args, 'fusion_mode', 'fixed_add'),
            'trajectory_residual_init': getattr(args, 'trajectory_residual_init', 0.25),
        }
        if encoder_class is TraceRepresentationControlledEncoder:
            encoder_kwargs.update(
                {
                    'trajectory_order_mode': intervention.get(
                        'trajectory_order_mode', 'original'
                    ),
                    'trajectory_time_mode': intervention.get(
                        'trajectory_time_mode', 'original'
                    ),
                    'static_feature_group_mask': intervention.get(
                        'static_feature_group_mask', 'none'
                    ),
                }
            )
        encoder = encoder_class(**encoder_kwargs)
        head_dim = args.d_model
    elif args.model == 'tlstm':
        encoder = ClinicalTLSTMEncoder(
            num_types=num_types,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pooling=args.pooling,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.hidden_dim
    elif args.model == 'retain':
        encoder = ClinicalRETAINEncoder(
            num_types=num_types,
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            time_scale_hours=args.time_scale_hours,
        )
        head_dim = args.d_model
    elif args.model in {'mhp', 'mhp_pretrained'}:
        encoder = MHPEncoderAdapter(
            num_types=num_types,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.dropout,
            pooling=args.pooling,
        )
        head_dim = args.d_model
    else:
        encoder = AMHPEncoderAdapter(
            num_types=num_types,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.dropout,
            pooling=args.pooling,
        )
        head_dim = args.d_model
    if args.model == 'tracehr' and args.trace_upgrade_variant == 'u4_delta':
        head = BoundedDeltaEndpointHead(
            input_dim=args.d_model,
            num_tasks=num_tasks,
            dropout=args.dropout,
            delta_bound=args.trace_delta_logit_bound,
        )
    elif getattr(encoder, 'task_conditioned_output', False):
        head = EndpointConditionedMultiTaskHead(
            input_dim=head_dim,
            num_tasks=num_tasks,
            dropout=args.dropout,
        )
    else:
        head = BinaryMultiTaskHead(input_dim=head_dim, num_tasks=num_tasks, dropout=args.dropout)
    return ClinicalPredictionModel(encoder, head)


def reproduction_metadata(model_name, baseline_stage='static_reproduction', hera_intervention=None):
    direct = {'gru', 'rnn', 'lstm', 'mlp', 'resmlp', 'rescnn', 'transformer', 'mamba'}
    clinical_adapters_2024plus = {
        'xtsformer',
        's2p2',
        'decoupled_mtpp_ode',
        'ehrmamba',
        'core_behrt',
        'hymate',
    }
    encoder_adaptation = {'mhp', 'amhp'}
    adapted = {'tlstm', 'retain'}
    engineering_transfer = {'mhp_pretrained', 'amhp_pretrained'}
    if model_name == 'tracehr':
        level = 'main_method_tracehr_dual_path_multiscale_relational_evidence'
        note = (
            'TRACEHR combines the paper-aligned order-sensitive Mamba trajectory with '
            'causal multi-scale, endpoint-conditioned relational evidence representations.'
        )
        original_input_protocol = 'Exp2/Exp4 fixed event-token ICU landmark PKL contract'
        original_objective = 'new main method; no paper-faithful baseline objective'
        current_objective = 'multi-task weighted BCE on post-landmark clinical labels'
        strict_loss_match = None
        paperfaithful_status = 'main_method_not_paperfaithful_baseline'
        table_role = 'main_method'
    elif model_name == 'hawkes_mamba' and hera_intervention == 'clean_core':
        level = 'main_method_trace_mamba_clean_core'
        note = (
            'TRACE-Mamba jointly models the temporal trajectory and landmark-conditioned '
            'clinical state without decayed-history memory or ranking supervision.'
        )
        original_input_protocol = 'Exp2/Exp4 fixed event-token ICU landmark PKL contract'
        original_objective = 'new main method; no paper-faithful baseline objective'
        current_objective = 'multi-task weighted BCE on post-landmark clinical labels'
        strict_loss_match = None
        paperfaithful_status = 'main_method_not_paperfaithful_baseline'
        table_role = 'main_method'
    elif model_name == 'hawkes_mamba' and is_trace_clean_core_intervention(hera_intervention):
        level = 'controlled_trace_mamba_clean_core_ablation'
        note = (
            'Controlled seed-screening ablation of the TRACE-Mamba temporal-trajectory '
            'and landmark-conditioned clinical-state core.'
        )
        original_input_protocol = 'Exp2/Exp4 fixed event-token ICU landmark PKL contract'
        original_objective = 'clean TRACE-Mamba core with one structural intervention'
        current_objective = 'multi-task weighted BCE on post-landmark clinical labels'
        strict_loss_match = None
        paperfaithful_status = 'controlled_main_method_ablation'
        table_role = 'controlled_ablation'
    elif model_name == 'hawkes_mamba' and is_trace_representation_intervention(hera_intervention):
        level = 'controlled_trace_mamba_representation_ablation'
        note = (
            'Single-factor control of the frozen TRACE-Mamba representation contract; '
            'all non-target trajectory, state, loss, and inference settings remain enabled.'
        )
        original_input_protocol = 'Exp2/Exp4 fixed event-token ICU landmark PKL contract'
        original_objective = 'frozen TRACE-Mamba Full with one representation intervention'
        current_objective = 'multi-task weighted BCE plus the nominal mortality ranking term'
        strict_loss_match = None
        paperfaithful_status = 'controlled_main_method_ablation'
        table_role = 'controlled_ablation'
    elif model_name == 'hawkes_mamba':
        level = 'main_method_rank_aware_hawkes_mamba'
        note = 'Learned exponential Hawkes-style event context is fused into a Mamba encoder for clinical risk ranking.'
        original_input_protocol = 'Exp2/Exp4 fixed event-token ICU landmark PKL contract'
        original_objective = 'new main method; no paper-faithful baseline objective'
        current_objective = 'multi-task weighted BCE with optional pairwise ranking over post-landmark clinical labels'
        strict_loss_match = None
        paperfaithful_status = 'main_method_not_paperfaithful_baseline'
        table_role = 'main_method'
    elif model_name in direct:
        level = 'fair_protocol_compatible_architecture'
        note = 'Architecture is evaluated directly on the fixed Experiment 2 event-sequence contract.'
        original_input_protocol = 'generic sequence architecture; no single original EHR cohort contract'
        original_objective = 'task-dependent supervised sequence objective in the downstream paper/task'
        current_objective = 'multi-task weighted BCE on post-landmark clinical labels'
        strict_loss_match = None
        paperfaithful_status = 'direct_supervised_baseline'
        table_role = 'main_table'
    elif model_name in clinical_adapters_2024plus:
        level = 'paper_source_2024plus_clinical_adapter'
        note = 'Recent event-sequence architecture is reproduced as an Exp2/Exp4 clinical-label adapter.'
        original_input_protocol = 'irregular marked event sequence with continuous-time modeling'
        original_objective = 'paper-specific event prediction or temporal point-process objective'
        current_objective = 'multi-task weighted BCE on post-landmark clinical labels'
        strict_loss_match = False
        paperfaithful_status = 'architecture_faithful_clinical_adapter_not_original_loss'
        table_role = '2024plus_adapter_main_or_appendix'
    elif model_name in encoder_adaptation:
        level = 'experiment2_encoder_adaptation'
        note = 'TPP-aware encoder adaptation with a supervised clinical prediction head; not the original TPP prediction head.'
        original_input_protocol = 'marked temporal event sequence'
        original_objective = 'temporal point-process event log-likelihood / next-event type-time objective'
        current_objective = 'multi-task weighted BCE on post-landmark clinical labels'
        strict_loss_match = False
        paperfaithful_status = 'clinical_adapter_not_original_tpp_loss'
        table_role = 'adapter_table_or_clearly_labeled_main_row'
    elif model_name in adapted:
        level = 'event_level_protocol_adaptation'
        note = 'Original method uses a different EHR representation; this run is an explicitly labeled event-level adaptation.'
        original_input_protocol = 'visit-level or variable-level EHR sequence in the original method'
        original_objective = 'supervised clinical prediction loss in the original method family'
        current_objective = 'multi-task weighted BCE on event-token ICU landmark samples'
        strict_loss_match = True
        paperfaithful_status = 'supervised_loss_match_but_input_protocol_adapted'
        table_role = 'secondary_adapter_table'
    elif model_name in engineering_transfer:
        level = 'engineering_transfer_only'
        note = 'Checkpoint transfer is not a formal pretraining-benefit result unless a compatible full TPP checkpoint is loaded.'
        original_input_protocol = 'marked temporal event sequence'
        original_objective = 'temporal point-process event log-likelihood / next-event type-time objective'
        current_objective = 'multi-task weighted BCE on post-landmark clinical labels after optional checkpoint transfer'
        strict_loss_match = False
        paperfaithful_status = 'engineering_transfer_not_paperfaithful_baseline'
        table_role = 'diagnostic_only'
    else:
        level = 'unspecified'
        note = ''
        original_input_protocol = 'unspecified'
        original_objective = 'unspecified'
        current_objective = 'unspecified'
        strict_loss_match = None
        paperfaithful_status = 'unspecified'
        table_role = 'unspecified'
    return {
        'baseline_stage': baseline_stage,
        'reproduction_level': level,
        'reproduction_note': note,
        'original_input_protocol': original_input_protocol,
        'original_training_objective': original_objective,
        'current_input_protocol': 'Exp2/Exp4 fixed event-token ICU landmark PKL contract',
        'current_training_objective': current_objective,
        'strict_loss_match_to_original': strict_loss_match,
        'paperfaithful_status': paperfaithful_status,
        'recommended_table_role': table_role,
    }


def checkpoint_candidates(model_name):
    # The old mixed-event Exp1 checkpoints are obsolete after the disease-set
    # protocol change. Pretraining transfer must be explicit until a compatible
    # disease-set Hawkes-Mamba checkpoint exists.
    return []


def extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(hasattr(value, 'shape') for value in payload.values()):
            return payload
    return {}


def candidate_target_keys(source_key):
    key = source_key.replace('module.', '')
    keys = [key, f'encoder.{key}']
    if key.startswith('encoder.'):
        keys.append(f'encoder.{key}')
    if key.startswith('module.encoder.'):
        keys.append('encoder.' + key[len('module.'):])
    return keys


def load_pretrained_if_requested(model, args, output_dir):
    if args.model not in {'mhp_pretrained', 'amhp_pretrained'}:
        return None
    checkpoint_path = Path(args.pretrained_checkpoint) if args.pretrained_checkpoint else None
    if checkpoint_path is None and args.auto_find_pretrained:
        candidates = checkpoint_candidates(args.model)
        checkpoint_path = candidates[0] if candidates else None
    report = {
        'checkpoint_path': None if checkpoint_path is None else str(checkpoint_path),
        'status': 'skipped',
        'loaded_keys': [],
        'skipped_keys': [],
        'missing_keys': [],
        'unexpected_keys': [],
        'embedding_compatible': False,
        'notes': '',
    }
    if checkpoint_path is None or not checkpoint_path.exists():
        report['notes'] = 'No compatible checkpoint candidate was found.'
        write_load_report(output_dir / 'load_report.md', report)
        return report
    payload = torch.load(checkpoint_path, map_location='cpu')
    source = extract_state_dict(payload)
    target = model.state_dict()
    update = {}
    skipped = []
    for source_key, tensor in source.items():
        matched = None
        for target_key in candidate_target_keys(source_key):
            if target_key in target and tuple(target[target_key].shape) == tuple(tensor.shape):
                matched = target_key
                break
        if matched is None:
            skipped.append(source_key)
        else:
            update[matched] = tensor
    current = model.state_dict()
    current.update(update)
    model.load_state_dict(current, strict=False)
    report['loaded_keys'] = sorted(update)
    report['skipped_keys'] = sorted(skipped)
    report['missing_keys'] = sorted(set(target) - set(update))
    report['unexpected_keys'] = sorted(set(source) - set(skipped))
    report['embedding_compatible'] = any(key.endswith('event_emb.weight') for key in update)
    report['status'] = 'loaded' if update and not skipped else ('partial_loaded' if update else 'skipped')
    report['notes'] = 'Partial loading is expected when Experiment 1 TPP heads or vocab embeddings differ.'
    write_load_report(output_dir / 'load_report.md', report)
    return report


def write_load_report(path, report):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('# TPP Pretraining Load Report\n\n')
        handle.write(f'* checkpoint path: {report.get("checkpoint_path")}\n')
        handle.write(f'* status: {report.get("status")}\n')
        handle.write(f'* embedding compatible: {report.get("embedding_compatible")}\n')
        handle.write(f'* loaded keys: {len(report.get("loaded_keys", []))}\n')
        handle.write(f'* skipped keys: {len(report.get("skipped_keys", []))}\n')
        handle.write(f'* missing keys: {len(report.get("missing_keys", []))}\n')
        handle.write(f'* unexpected keys: {len(report.get("unexpected_keys", []))}\n')
        handle.write(f'* notes: {report.get("notes")}\n')
        handle.write('\n## Loaded Key Examples\n\n')
        for key in report.get('loaded_keys', [])[:50]:
            handle.write(f'* {key}\n')
        handle.write('\n## Skipped Key Examples\n\n')
        for key in report.get('skipped_keys', [])[:50]:
            handle.write(f'* {key}\n')


def batch_events(batch):
    return int(batch['event_type'].ne(0).sum().item())


def global_grad_norm(parameters):
    norms = []
    for parameter in parameters:
        if parameter.grad is not None:
            norms.append(parameter.grad.detach().data.norm(2).float())
    if not norms:
        return None
    return float(torch.norm(torch.stack(norms), 2).detach().cpu())


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    pos_weight,
    grad_clip=0.0,
    epoch=None,
    log_interval=100,
    progress='log',
    ranking_loss_weight=0.0,
    ranking_objective_gate=1.0,
    ranking_task_indices=None,
    ranking_max_pairs_per_task=4096,
):
    model.train()
    total_loss = 0.0
    total_bce_loss = 0.0
    total_ranking_loss = 0.0
    total_samples = 0
    total_events = 0
    total_ranking_pairs = 0
    grad_norms = []
    started = time.perf_counter()
    total_batches = len(loader)
    show_logs = progress in {'log', 'both'}
    show_tqdm = progress in {'tqdm', 'both'} and tqdm is not None
    if progress in {'tqdm', 'both'} and tqdm is None:
        log_message('WARN', 'tqdm is unavailable; falling back to structured progress logs')
        show_logs = True

    progress_bar = None
    if show_tqdm:
        progress_bar = tqdm(
            total=total_batches,
            desc=f'epoch {epoch}',
            unit='batch',
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )
    try:
        for batch_index, batch in enumerate(loader, start=1):
            event_type = batch['event_type'].to(device)
            event_time = batch['time_since_start'].to(device)
            time_gap = batch['time_since_last_event'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device)
            source_features = batch.get('source_features')
            if source_features is not None:
                source_features = source_features.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(
                event_type,
                event_time,
                time_gap=time_gap,
                source_features=source_features,
            )
            bce_loss = masked_bce_with_logits(logits, labels, label_mask, pos_weight=pos_weight)
            ranking_loss = logits.new_tensor(0.0)
            ranking_stats = {'pair_count': 0}
            if float(ranking_loss_weight) * float(ranking_objective_gate) > 0:
                ranking_loss, ranking_stats = pairwise_logistic_ranking_loss(
                    logits,
                    labels,
                    label_mask=label_mask,
                    task_indices=ranking_task_indices,
                    max_pairs_per_task=ranking_max_pairs_per_task,
                )
            loss = combine_hera_losses(
                bce_loss,
                ranking_loss,
                ranking_loss_weight,
                ranking_objective_gate,
            )
            loss.backward()
            norm = global_grad_norm(model.parameters())
            if norm is not None:
                grad_norms.append(norm)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_bce_loss += float(bce_loss.detach().cpu()) * batch_size
            total_ranking_loss += float(ranking_loss.detach().cpu()) * batch_size
            total_samples += batch_size
            total_events += batch_events(batch)
            total_ranking_pairs += int(ranking_stats.get('pair_count', 0))
            should_report = log_interval and (
                batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches
            )
            if progress_bar is not None:
                progress_bar.update(1)
                if should_report:
                    elapsed = max(time.perf_counter() - started, 1e-8)
                    progress_bar.set_postfix(
                        loss=f'{total_loss / max(total_samples, 1):.4f}',
                        events=total_events,
                        eps=f'{total_events / elapsed:.0f}',
                    )
            if show_logs and should_report:
                elapsed = max(time.perf_counter() - started, 1e-8)
                log_message(
                    'PROGRESS',
                    f'epoch={epoch} batch={batch_index}/{total_batches} '
                    f'samples={total_samples} events={total_events} '
                    f'avg_loss={total_loss / max(total_samples, 1):.6f} '
                    f'avg_bce={total_bce_loss / max(total_samples, 1):.6f} '
                    f'avg_rank={total_ranking_loss / max(total_samples, 1):.6f} '
                    f'ranking_pairs={total_ranking_pairs} '
                    f'events_per_sec={total_events / elapsed:.1f}',
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    elapsed = time.perf_counter() - started
    grad_array = np.asarray(grad_norms, dtype=float)
    return {
        'loss': total_loss / max(total_samples, 1),
        'bce_loss': total_bce_loss / max(total_samples, 1),
        'ranking_loss': total_ranking_loss / max(total_samples, 1),
        'ranking_pairs': total_ranking_pairs,
        'events': total_events,
        'seconds': elapsed,
        'events_per_sec': total_events / elapsed if elapsed > 0 else None,
        'grad_norm_median': float(np.median(grad_array)) if grad_array.size else None,
        'grad_norm_mean': float(np.mean(grad_array)) if grad_array.size else None,
        'grad_norm_max': float(np.max(grad_array)) if grad_array.size else None,
    }


@torch.no_grad()
def predict(model, loader, device, tasks):
    model.eval()
    probs, labels, ids = [], [], []
    total_events = 0
    started = time.perf_counter()
    for batch in loader:
        event_type = batch['event_type'].to(device)
        event_time = batch['time_since_start'].to(device)
        time_gap = batch['time_since_last_event'].to(device)
        source_features = batch.get('source_features')
        if source_features is not None:
            source_features = source_features.to(device)
        logits, _, _ = model(
            event_type,
            event_time,
            time_gap=time_gap,
            source_features=source_features,
        )
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(batch['labels'].detach().cpu().numpy())
        ids.append(torch.stack([batch['subject_id'], batch['hadm_id'], batch['stay_id']], dim=1).detach().cpu().numpy())
        total_events += batch_events(batch)
    elapsed = time.perf_counter() - started
    if probs:
        probs = np.concatenate(probs, axis=0)
        labels = np.concatenate(labels, axis=0)
        ids = np.concatenate(ids, axis=0)
    else:
        probs = np.zeros((0, len(tasks)), dtype=np.float32)
        labels = np.zeros((0, len(tasks)), dtype=np.float32)
        ids = np.zeros((0, 3), dtype=np.int64)
    return probs, labels, ids, {'inference_time_sec': elapsed, 'events': total_events, 'events_per_sec': total_events / elapsed if elapsed > 0 else None}


def safe_stats(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    return {
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
    }


def find_embedding_stats(model):
    for module in model.modules():
        event_emb = getattr(module, 'event_emb', None)
        weight = getattr(event_emb, 'weight', None)
        if weight is not None:
            norms = weight.detach().float().norm(dim=1).cpu().numpy()
            return safe_stats(norms)
    return {'mean': None, 'std': None, 'min': None, 'max': None}


def source_counts_from_samples(samples):
    counts = {}
    for sample in samples:
        for source, value in sample.get('source_counts', {}).items():
            counts[source] = counts.get(source, 0) + int(value)
    return counts


@torch.no_grad()
def collect_diagnostic_stats(model, loaders, samples_by_split, device):
    model.eval()
    split_stats = {}
    for split, loader in loaders.items():
        hidden_norms = []
        pooled_norms = []
        logits_values = []
        probabilities = []
        for batch in loader:
            event_type = batch['event_type'].to(device)
            event_time = batch['time_since_start'].to(device)
            time_gap = batch['time_since_last_event'].to(device)
            source_features = batch.get('source_features')
            if source_features is not None:
                source_features = source_features.to(device)
            logits, sequence_hidden, pooled = model(
                event_type,
                event_time,
                time_gap=time_gap,
                source_features=source_features,
            )
            non_pad = event_type.ne(0)
            if non_pad.any():
                hidden_norms.extend(sequence_hidden[non_pad].detach().float().norm(dim=-1).cpu().numpy().tolist())
            pooled_values = pooled.detach().float().reshape(-1, pooled.size(-1))
            pooled_norms.extend(pooled_values.norm(dim=-1).cpu().numpy().tolist())
            logits_values.extend(logits.detach().float().reshape(-1).cpu().numpy().tolist())
            probabilities.extend(torch.sigmoid(logits).detach().float().reshape(-1).cpu().numpy().tolist())
        probs_array = np.asarray(probabilities, dtype=float)
        split_stats[split] = {
            'hidden_norm': safe_stats(hidden_norms),
            'pooled_norm': safe_stats(pooled_norms),
            'logit': safe_stats(logits_values),
            'probability': safe_stats(probabilities),
            'prob_saturation_low': float(np.mean(probs_array < 0.01)) if probs_array.size else None,
            'prob_saturation_high': float(np.mean(probs_array > 0.99)) if probs_array.size else None,
            'source_counts': source_counts_from_samples(samples_by_split.get(split, [])),
        }
    return {
        'splits': split_stats,
        'embedding_norm': find_embedding_stats(model),
    }


def monitor_value(metrics, monitor):
    return metrics.get('aggregate', {}).get(monitor)


def get_git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def get_source_hashes():
    root = Path(__file__).resolve().parent
    relative_paths = (
        'train_exp2_clinical_deep.py',
        'models/clinical_tracehr.py',
        'models/clinical_hawkes_mamba.py',
        'models/clinical_heads.py',
        'preprocess/ClinicalDataset.py',
        'Experiments/scripts/exp2c_metrics.py',
    )
    hashes = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            hashes[relative] = None
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return hashes


def main(initial_checkpoint_callback=None):
    args = parse_args()
    validate_stage_objective(args)
    hera_contract = resolve_hera_training_contract(args)
    intervention_audit = intervention_audit_from_args(args)
    source_hashes = get_source_hashes()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_stage('Experiment 2 clinical deep training: initialization')
    log_message('INFO', f'model={args.model} seed={args.seed} requested_device={args.device}')
    log_message('INFO', f'data_dir={args.data_dir}')
    log_message('INFO', f'output_dir={output_dir}')
    log_message(
        'INFO',
        f'max_epochs={args.max_epochs} patience={args.early_stopping_patience} '
        f'batch_size={args.batch_size} lr={args.lr} weight_decay={args.weight_decay} '
        f'pooling={args.pooling} log_interval={args.log_interval} '
        f'baseline_stage={args.baseline_stage}',
    )

    log_stage('Load fixed Experiment 2 data splits')
    samples, tasks, num_types = load_splits(args.data_dir, args.evaluation_scope)
    samples = limit_samples_for_smoke(samples, args.max_train_samples, args.max_eval_samples)
    primary_tasks = args.primary_tasks or tasks
    effective_ranking_weight = float(args.ranking_loss_weight) * float(hera_contract['ranking_objective_gate'])
    if effective_ranking_weight > 0:
        ranking_task_indices, ranking_tasks = resolve_task_indices(tasks, args.ranking_tasks or primary_tasks)
    else:
        ranking_task_indices = []
        ranking_tasks = []
    training_objective = training_objective_description(args, ranking_tasks)
    args.time_scale_hours = resolve_time_scale_hours(args.data_dir, args.time_scale_hours)
    source_names = []
    if args.model == 'tracehr' or (
        args.model == 'hawkes_mamba'
        and args.static_context_scale > 0
        and args.static_feature_set == 'rich'
        and float(hera_contract['rich_static_fusion_gate']) != 0.0
    ):
        source_names = infer_source_names(samples['train'])
    args.static_source_names = list(source_names)
    args.static_source_dim = len(source_names) * 2
    model_config = model_config_from_args(args)
    determinism_config = determinism_config_from_args(args)
    device, device_message = resolve_device(args.device)
    for split in samples:
        event_count = sum(len(sample['type_event']) for sample in samples[split])
        log_message(
            'INFO',
            f'split={split} samples={len(samples[split])} events={event_count} '
            f'label_prevalence={label_prevalence(samples[split], tasks)}',
        )
    log_message('INFO', f'tasks={tasks}')
    log_message('INFO', f'primary_tasks={primary_tasks}')
    log_message('INFO', f'training_objective={training_objective}')
    log_message('INFO', f'ranking_tasks={ranking_tasks} ranking_task_indices={ranking_task_indices}')
    log_message('INFO', f'static_source_names={source_names} static_source_dim={args.static_source_dim}')
    if args.model == 'hawkes_mamba':
        log_message(
            'INFO',
            f'hera_intervention={getattr(args, "hera_intervention", None) or "full"} '
            f'contract={hera_contract}',
        )
    elif args.model == 'tracehr':
        log_message(
            'INFO',
            'tracehr_configuration='
            f'{intervention_audit}',
        )
    log_message('INFO', f'intervention_audit={intervention_audit}')
    log_message('INFO', f'model_config={model_config}')
    log_message('INFO', f'determinism_config={determinism_config}')
    log_message('INFO', f'num_event_types_raw={num_types} time_scale_hours={args.time_scale_hours}')
    log_message('INFO', f'resolved_device={device} device_message={device_message}')

    if device.type == 'cuda':
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats()

    log_stage('Build dataloaders and model')
    split_seed_offsets = {'train': 0, 'dev': 100_000, 'test': 200_000}
    loaders = {
        split: get_clinical_dataloader(
            samples[split],
            tasks,
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            seed=args.seed + split_seed_offsets[split],
            source_names=source_names,
        )
        for split in samples
    }
    model = build_model(args, num_types, len(tasks)).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    log_message(
        'INFO',
        f'model={args.model} reproduction={reproduction_metadata(args.model, args.baseline_stage, getattr(args, "hera_intervention", None))["reproduction_level"]} '
        f'total_parameters={total_parameters} trainable_parameters={trainable_parameters}',
    )
    model_reproduction = reproduction_metadata(
        args.model,
        args.baseline_stage,
        getattr(args, 'hera_intervention', None),
    )
    model_reproduction['current_training_objective'] = training_objective
    if effective_ranking_weight > 0:
        model_reproduction['loss_family'] = 'weighted_bce_plus_pairwise_logistic_ranking'
    elif args.ranking_loss_weight > 0 and not is_trace_clean_core(args):
        model_reproduction['loss_family'] = 'weighted_bce_plus_gated_zero_pairwise_logistic_ranking'
    else:
        model_reproduction['loss_family'] = 'weighted_bce'
    log_message(
        'INFO',
        f'paperfaithful_status={model_reproduction["paperfaithful_status"]} '
        f'strict_loss_match_to_original={model_reproduction["strict_loss_match_to_original"]} '
        f'recommended_table_role={model_reproduction["recommended_table_role"]}',
    )
    log_message(
        'INFO',
        'dataloader_batches '
        + ' '.join(f'{split_name}={len(loaders[split_name])}' for split_name in loaders)
        + f' num_workers={args.num_workers} pin_memory={args.pin_memory}',
    )
    load_report = load_pretrained_if_requested(model, args, output_dir)
    if load_report is not None:
        log_message(
            'INFO',
            f'pretraining_status={load_report.get("status")} loaded_keys={len(load_report.get("loaded_keys", []))} '
            f'skipped_keys={len(load_report.get("skipped_keys", []))}',
        )
    pos_weight = compute_pos_weight(samples['train'], tasks, device)
    log_message('INFO', f'train_only_pos_weight={[float(value) for value in pos_weight.detach().cpu()]}')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_rows = []
    best_value = -float('inf')
    best_epoch = 0
    best_state = None
    best_dev_auprc = {'epoch': None, 'value': None}
    best_dev_auroc = {'epoch': None, 'value': None}
    no_improve = 0
    train_started = time.perf_counter()

    # Optional dev-only epoch-zero candidate for a frozen pretrained base.
    # Ordinary historical training has no callback and is unchanged.
    if initial_checkpoint_callback is not None:
        initial_metrics = initial_checkpoint_callback(model, loaders['dev'], device, tasks, primary_tasks)
        initial_value = monitor_value(initial_metrics, args.monitor)
        if initial_value is None or not np.isfinite(initial_value):
            raise ValueError('Invalid epoch-zero monitor')
        best_value = float(initial_value)
        best_state = copy.deepcopy(model.state_dict())
        best_dev_auprc = {'epoch': 0, 'value': initial_metrics['aggregate']['mean_primary_auprc']}
        best_dev_auroc = {'epoch': 0, 'value': initial_metrics['aggregate']['mean_primary_auroc']}
        torch.save({'model_state_dict': best_state, 'epoch': 0, 'tasks': tasks,
                    'num_event_types_raw': num_types, 'config': vars(args)}, ckpt_dir / 'best.pt')
        save_json(output_dir / 'epoch_zero_metrics.json', initial_metrics)

    log_stage('Train with dev-only early stopping')
    for epoch in range(1, args.max_epochs + 1):
        log_message('EPOCH', f'BEGIN epoch={epoch}/{args.max_epochs}', blank_before=True)
        train_stats = train_epoch(
            model,
            loaders['train'],
            optimizer,
            device,
            pos_weight,
            grad_clip=args.grad_clip,
            epoch=epoch,
            log_interval=args.log_interval,
            progress=args.progress,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_objective_gate=hera_contract['ranking_objective_gate'],
            ranking_task_indices=ranking_task_indices,
            ranking_max_pairs_per_task=args.ranking_max_pairs_per_task,
        )
        log_message(
            'INFO',
            f'epoch={epoch} train_complete loss={train_stats["loss"]:.6f} '
            f'bce_loss={train_stats["bce_loss"]:.6f} '
            f'ranking_loss={train_stats["ranking_loss"]:.6f} '
            f'ranking_pairs={train_stats["ranking_pairs"]} '
            f'events={train_stats["events"]} seconds={train_stats["seconds"]:.2f} '
            f'events_per_sec={train_stats["events_per_sec"]:.1f}',
        )
        log_message('INFO', f'epoch={epoch} evaluating_dev split=dev')
        dev_probs, dev_labels, _, dev_eff = predict(model, loaders['dev'], device, tasks)
        dev_metrics = evaluate_multitask(dev_labels, dev_probs, tasks, primary_tasks=primary_tasks)
        dev_auprc = dev_metrics['aggregate'].get('mean_primary_auprc')
        dev_auroc = dev_metrics['aggregate'].get('mean_primary_auroc')
        if dev_auprc is not None and (best_dev_auprc['value'] is None or dev_auprc > best_dev_auprc['value']):
            best_dev_auprc = {'epoch': epoch, 'value': dev_auprc}
        if dev_auroc is not None and (best_dev_auroc['value'] is None or dev_auroc > best_dev_auroc['value']):
            best_dev_auroc = {'epoch': epoch, 'value': dev_auroc}
        value = monitor_value(dev_metrics, args.monitor)
        improved = value is not None and value > best_value
        if improved:
            best_value = float(value)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({'model_state_dict': best_state, 'epoch': epoch, 'tasks': tasks, 'num_event_types_raw': num_types, 'config': vars(args)}, ckpt_dir / 'best.pt')
            no_improve = 0
            log_message('CHECKPOINT', f'saved_best epoch={epoch} monitor={args.monitor} value={best_value:.6f} path={ckpt_dir / "best.pt"}')
        else:
            no_improve += 1
        row = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'train_bce_loss': train_stats['bce_loss'],
            'train_ranking_loss': train_stats['ranking_loss'],
            'train_ranking_pairs': train_stats['ranking_pairs'],
            'train_events_per_sec': train_stats['events_per_sec'],
            'dev_mean_primary_auroc': dev_auroc,
            'dev_mean_primary_auprc': dev_auprc,
            'monitor': args.monitor,
            'monitor_value': value,
            'improved': improved,
            'grad_norm_median': train_stats['grad_norm_median'],
            'grad_norm_mean': train_stats['grad_norm_mean'],
            'grad_norm_max': train_stats['grad_norm_max'],
        }
        train_rows.append(row)
        log_message(
            'RESULT',
            f'END epoch={epoch}/{args.max_epochs} train_loss={train_stats["loss"]:.6f} '
            f'train_bce_loss={train_stats["bce_loss"]:.6f} '
            f'train_ranking_loss={train_stats["ranking_loss"]:.6f} '
            f'dev_mean_primary_auprc={dev_auprc} dev_mean_primary_auroc={dev_auroc} '
            f'grad_norm_median={train_stats["grad_norm_median"]} improved={improved} '
            f'no_improve={no_improve}/{args.early_stopping_patience}',
            blank_after=True,
        )
        if no_improve >= args.early_stopping_patience:
            log_message(
                'EARLY_STOP',
                f'triggered at epoch={epoch}; best_epoch={best_epoch} best_{args.monitor}={best_value}',
                blank_after=True,
            )
            break
    train_time = time.perf_counter() - train_started
    if best_state is not None:
        model.load_state_dict(best_state)

    log_stage('Final evaluation from best dev checkpoint')
    log_message('INFO', f'best_epoch={best_epoch} best_monitor={args.monitor} best_monitor_value={best_value}')
    log_message('INFO', 'evaluating_dev_for_threshold_selection=true')
    dev_probs, dev_labels, dev_ids, dev_eff = predict(model, loaders['dev'], device, tasks)
    test_prediction = None
    if args.evaluation_scope == 'dev_and_test':
        log_message('INFO', 'evaluating_test_once=true')
        test_prediction = predict(model, loaders['test'], device, tasks)
    else:
        log_message('INFO', 'held_out_test_loaded=false held_out_test_evaluated=false')
    diagnostic_stats = collect_diagnostic_stats(model, loaders, samples, device) if args.save_diagnostic_stats else None
    thresholds = select_thresholds_on_dev(dev_labels, dev_probs, tasks, metric='f1')
    log_message('INFO', f'dev_selected_thresholds={thresholds}')
    split_metrics = {
        'dev': evaluate_multitask(
            dev_labels,
            dev_probs,
            tasks,
            thresholds=thresholds,
            primary_tasks=primary_tasks,
        )
    }
    efficiency = {'dev': dev_eff}
    prediction_artifacts = [('dev', dev_probs, dev_labels, dev_ids)]
    if test_prediction is not None:
        test_probs, test_labels, test_ids, test_eff = test_prediction
        split_metrics['test'] = evaluate_multitask(
            test_labels,
            test_probs,
            tasks,
            thresholds=thresholds,
            primary_tasks=primary_tasks,
        )
        efficiency['test'] = test_eff
        prediction_artifacts.append(('test', test_probs, test_labels, test_ids))
    metrics = {
        'model': args.model,
        **model_reproduction,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'data_dir': str(args.data_dir),
        'tasks': tasks,
        'primary_tasks': primary_tasks,
        'seed': args.seed,
        'evaluation_scope': args.evaluation_scope,
        'training_objective': training_objective,
        'intervention_audit': intervention_audit,
        'model_config': model_config,
        'determinism_config': determinism_config,
        'source_hashes': source_hashes,
        'loss_config': {
            'ranking_loss_weight': float(args.ranking_loss_weight),
            'ranking_objective_gate': float(hera_contract['ranking_objective_gate']),
            'effective_ranking_loss_weight': effective_ranking_weight,
            'ranking_tasks': ranking_tasks,
            'ranking_task_indices': ranking_task_indices,
            'ranking_max_pairs_per_task': int(args.ranking_max_pairs_per_task),
        },
        'num_event_types_raw': num_types,
        'num_event_types_with_padding': num_types + 1,
        'best_epoch': best_epoch,
        'best_monitor': args.monitor,
        'best_monitor_value': None if best_value == -float('inf') else best_value,
        'best_dev_auprc': best_dev_auprc,
        'best_dev_auroc': best_dev_auroc,
        'train_time_sec': train_time,
        'train_epochs_ran': len(train_rows),
        'peak_gpu_memory_mb': float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == 'cuda' else None,
        'device': str(device),
        'device_message': device_message,
        'pos_weight': [float(value) for value in pos_weight.detach().cpu()],
        'pretraining_load': load_report,
        'splits': split_metrics,
        'efficiency': efficiency,
    }
    if diagnostic_stats is not None:
        save_json(output_dir / 'diagnostic_stats.json', diagnostic_stats)
    save_json(output_dir / 'metrics.json', metrics)
    save_json(output_dir / 'selected_thresholds.json', thresholds)
    with open(output_dir / 'train_log.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(train_rows[0].keys()) if train_rows else ['epoch'])
        writer.writeheader()
        writer.writerows(train_rows)
    for split, probs, labels, ids in prediction_artifacts:
        np.savez_compressed(output_dir / f'predictions_{split}.npz', tasks=np.asarray(tasks, dtype=object), probs=probs, labels=labels, ids=ids)
    config = vars(args).copy()
    config.update({
        'resolved_device': str(device),
        'device_message': device_message,
        'pos_weight': [float(value) for value in pos_weight.detach().cpu()],
        'training_objective': training_objective,
        'model_config': model_config,
        'determinism_config': determinism_config,
        'resolved_ranking_tasks': ranking_tasks,
        'resolved_ranking_task_indices': ranking_task_indices,
    })
    with open(output_dir / 'config_snapshot.yaml', 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    save_json(
        output_dir / 'run_manifest.json',
        {
            'model': args.model,
            **model_reproduction,
            'data_dir': str(args.data_dir),
            'output_dir': str(output_dir),
            'seed': args.seed,
            'evaluation_scope': args.evaluation_scope,
            'training_objective': training_objective,
            'intervention_audit': intervention_audit,
            'model_config': model_config,
            'determinism_config': determinism_config,
            'source_hashes': source_hashes,
            'loss_config': {
                'ranking_loss_weight': float(args.ranking_loss_weight),
                'ranking_objective_gate': float(hera_contract['ranking_objective_gate']),
                'effective_ranking_loss_weight': effective_ranking_weight,
                'ranking_tasks': ranking_tasks,
                'ranking_task_indices': ranking_task_indices,
                'ranking_max_pairs_per_task': int(args.ranking_max_pairs_per_task),
            },
            'command': [sys.executable] + sys.argv,
            'git_commit': get_git_commit(),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'artifacts': {
                'checkpoint': str(ckpt_dir / 'best.pt'),
                'metrics': str(output_dir / 'metrics.json'),
                'train_log': str(output_dir / 'train_log.csv'),
                'predictions_dev': str(output_dir / 'predictions_dev.npz'),
                'predictions_test': (
                    str(output_dir / 'predictions_test.npz')
                    if args.evaluation_scope == 'dev_and_test'
                    else None
                ),
                'selected_thresholds': str(output_dir / 'selected_thresholds.json'),
                'diagnostic_stats': str(output_dir / 'diagnostic_stats.json') if diagnostic_stats is not None else None,
            },
        },
    )

    log_stage('Artifacts saved')
    log_message('ARTIFACT', f'metrics={output_dir / "metrics.json"}')
    log_message('ARTIFACT', f'checkpoint={ckpt_dir / "best.pt"}')
    log_message('ARTIFACT', f'train_log={output_dir / "train_log.csv"}')
    log_message('ARTIFACT', f'predictions_dev={output_dir / "predictions_dev.npz"}')
    if args.evaluation_scope == 'dev_and_test':
        log_message('ARTIFACT', f'predictions_test={output_dir / "predictions_test.npz"}')
    log_message(
        'COMPLETE',
        f'model={args.model} best_epoch={best_epoch} epochs_ran={len(train_rows)} '
        f'train_time_sec={train_time:.2f}',
        blank_after=True,
    )


if __name__ == '__main__':
    main()
