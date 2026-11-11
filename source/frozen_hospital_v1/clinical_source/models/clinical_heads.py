import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryMultiTaskHead(nn.Module):
    """Linear multi-task binary prediction head."""

    def __init__(self, input_dim, num_tasks, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_tasks),
        )

    def forward(self, pooled_hidden):
        return self.net(pooled_hidden)


class EndpointConditionedMultiTaskHead(nn.Module):
    """Apply one binary head to each endpoint-conditioned representation."""

    def __init__(self, input_dim, num_tasks, dropout=0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tasks = int(num_tasks)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(self.num_tasks, self.input_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_tasks))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, pooled_hidden):
        if pooled_hidden.ndim != 3:
            raise ValueError('endpoint-conditioned head expects [batch, task, feature] input')
        if pooled_hidden.size(1) != self.num_tasks or pooled_hidden.size(2) != self.input_dim:
            raise ValueError(
                f'expected [batch, {self.num_tasks}, {self.input_dim}], '
                f'got {tuple(pooled_hidden.shape)}'
            )
        value = self.dropout(pooled_hidden)
        return torch.einsum('btd,td->bt', value, self.weight) + self.bias


class BoundedDeltaEndpointHead(nn.Module):
    """Base endpoint logits plus a zero-initialized bounded evidence correction."""

    def __init__(self, input_dim, num_tasks, dropout=0.1, delta_bound=0.5):
        super().__init__()
        if float(delta_bound) <= 0:
            raise ValueError('delta_bound must be positive')
        self.input_dim = int(input_dim)
        self.num_tasks = int(num_tasks)
        self.delta_bound = float(delta_bound)
        self.dropout = nn.Dropout(dropout)
        self.base_weight = nn.Parameter(torch.empty(self.num_tasks, self.input_dim))
        self.base_bias = nn.Parameter(torch.zeros(self.num_tasks))
        self.delta_weight = nn.Parameter(torch.zeros(self.num_tasks, self.input_dim))
        self.delta_bias = nn.Parameter(torch.zeros(self.num_tasks))
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, pooled_hidden):
        if pooled_hidden.ndim != 3:
            raise ValueError('bounded-delta head expects [batch, task, feature] input')
        expected_width = 2 * self.input_dim
        if pooled_hidden.size(1) != self.num_tasks or pooled_hidden.size(2) != expected_width:
            raise ValueError(
                f'expected [batch, {self.num_tasks}, {expected_width}], '
                f'got {tuple(pooled_hidden.shape)}'
            )
        base, evidence = pooled_hidden.split(self.input_dim, dim=-1)
        base = self.dropout(base)
        evidence = self.dropout(evidence)
        base_logits = (
            torch.einsum('btd,td->bt', base, self.base_weight) + self.base_bias
        )
        delta_logits = (
            torch.einsum('btd,td->bt', evidence, self.delta_weight) + self.delta_bias
        )
        return base_logits + self.delta_bound * torch.tanh(delta_logits)


class ClinicalPredictionModel(nn.Module):
    """Encoder plus clinical prediction head."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        encoder_kwargs = {}
        if time_gap is not None and getattr(self.encoder, 'uses_time_gap', False):
            encoder_kwargs['time_gap'] = time_gap
        if source_features is not None and getattr(self.encoder, 'uses_source_features', False):
            encoder_kwargs['source_features'] = source_features
        pooled, sequence_hidden = self.encoder(event_type, event_time, **encoder_kwargs)
        logits = self.head(pooled)
        return logits, sequence_hidden, pooled


def masked_bce_with_logits(logits, labels, label_mask=None, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none', pos_weight=pos_weight)
    if label_mask is None:
        return loss.mean()
    loss = loss * label_mask
    denominator = label_mask.sum().clamp_min(1.0)
    return loss.sum() / denominator


def pairwise_logistic_ranking_loss(
    logits,
    labels,
    label_mask=None,
    task_indices=None,
    max_pairs_per_task=4096,
):
    """Pairwise logistic loss that ranks positive cases above negative cases."""

    if task_indices is None:
        task_indices = range(logits.size(1))
    losses = []
    task_pair_counts = {}
    total_pairs = 0
    for task_index in task_indices:
        task_index = int(task_index)
        valid = torch.ones_like(labels[:, task_index], dtype=torch.bool)
        if label_mask is not None:
            valid = label_mask[:, task_index].bool()
        task_labels = labels[:, task_index]
        task_logits = logits[:, task_index]
        positive = torch.nonzero(valid & (task_labels > 0.5), as_tuple=False).flatten()
        negative = torch.nonzero(valid & (task_labels <= 0.5), as_tuple=False).flatten()
        possible_pairs = int(positive.numel() * negative.numel())
        if possible_pairs == 0:
            task_pair_counts[str(task_index)] = 0
            continue
        max_pairs = int(max_pairs_per_task or 0)
        if max_pairs > 0 and possible_pairs > max_pairs:
            pos_index = positive[torch.randint(positive.numel(), (max_pairs,), device=logits.device)]
            neg_index = negative[torch.randint(negative.numel(), (max_pairs,), device=logits.device)]
            differences = task_logits[pos_index] - task_logits[neg_index]
            used_pairs = max_pairs
        else:
            differences = task_logits[positive].unsqueeze(1) - task_logits[negative].unsqueeze(0)
            used_pairs = possible_pairs
        losses.append(F.softplus(-differences).mean())
        task_pair_counts[str(task_index)] = used_pairs
        total_pairs += used_pairs
    if not losses:
        return logits.new_tensor(0.0), {'pair_count': 0, 'task_pair_counts': task_pair_counts}
    return torch.stack(losses).mean(), {'pair_count': total_pairs, 'task_pair_counts': task_pair_counts}
