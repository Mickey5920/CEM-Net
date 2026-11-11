"""Read-only loss-gradient summaries on the original training graph."""
import torch


def group(name):
    if name == 'base.input_members': return 'input_members'
    if name.startswith('base.numerical.'): return 'periodic_frontend'
    if name.startswith('base.'): return 'backbone_and_head'
    if name.startswith('fusion.auxiliary_head.'): return 'auxiliary_head'
    if name == 'fusion.trajectory.variable.weight': return 'shared_variable_embedding'
    if name.startswith('fusion.trajectory.'): return 'trajectory_encoder'
    if name.startswith('fusion.summary.'): return 'summary_encoder'
    if name.startswith(('fusion.refine.', 'fusion.gate.')): return 'summary_refinement'
    if name.startswith(('fusion.query.', 'fusion.key.', 'fusion.value.')): return 'attention'
    return 'fusion_readout'


def path_gradients(model, losses):
    pairs = list(model.named_parameters())
    parameters = [p for _, p in pairs]
    g0, ga = [torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
              for loss in losses]
    groups = {}
    for (name, p), main, aux in zip(pairs, g0, ga):
        if name.startswith('base.'):
            assert aux is None, ('Auxiliary depends on backbone', name)
        key = group(name)
        row = groups.setdefault(key, dict(tensors=0, main_connected=0, aux_connected=0,
                                         sums=p.new_zeros(3)))
        row['tensors'] += 1
        row['main_connected'] += main is not None
        row['aux_connected'] += aux is not None
        if main is not None:
            assert torch.isfinite(main).all(), name
            row['sums'][0] += main.square().sum()
        if aux is not None:
            assert torch.isfinite(aux).all(), name
            row['sums'][1] += aux.square().sum()
        if main is not None and aux is not None:
            row['sums'][2] += (main*aux).sum()
    for row in groups.values():
        main, aux, dot = row.pop('sums').detach().double().cpu().tolist()
        row.update(main_norm=main**.5, auxiliary_norm=aux**.5,
                   cosine=dot/(main*aux)**.5 if main>0 and aux>0 else None,
                   auxiliary_to_main_norm=(aux/main)**.5 if main>0 else None)
    assert groups['auxiliary_head']['auxiliary_norm']>0
    return dict(auxiliary_base_gradient_absent=True, training_graph=True,
                diagnostic_updates=0, microbatch='first original microbatch of checked update',
                groups=groups)
