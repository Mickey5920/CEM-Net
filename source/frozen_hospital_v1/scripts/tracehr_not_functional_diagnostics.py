"""Read-only checks distinguish removed supervision from a frozen/unused model."""
import torch
from scripts.tracehr_auxiliary_isolation_diagnostics import group


def path_gradients(model, losses):
    pairs = list(model.named_parameters())
    parameters = [p for _, p in pairs]
    main_loss, aux_loss = losses
    g0 = torch.autograd.grad(main_loss, parameters, allow_unused=True, retain_graph=True)
    ga = (torch.autograd.grad(aux_loss, parameters, allow_unused=True, retain_graph=True)
          if aux_loss is not None else [None]*len(parameters))
    assert (aux_loss is not None) == model.auxiliary_enabled
    groups = {}
    for (name, p), main, aux in zip(pairs, g0, ga):
        if name.startswith('base.'):
            assert aux is None, ('Auxiliary depends on backbone', name)
        if not model.main_residual_enabled and name.startswith('fusion.'):
            assert main is None, ('Removed main residual still carries gradient', name)
        assert main is not None or aux is not None, ('Retained tensor disconnected', name)
        row = groups.setdefault(group(name), dict(tensors=0, main_connected=0, aux_connected=0, sums=p.new_zeros(3)))
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
    if model.auxiliary_enabled:
        assert groups['auxiliary_head']['auxiliary_norm'] > 0
    else:
        assert 'auxiliary_head' not in groups and all(r['aux_connected']==0 for r in groups.values())
    return dict(auxiliary_base_gradient_absent=True, training_graph=True, diagnostic_updates=0,
                auxiliary_enabled=model.auxiliary_enabled, main_residual_enabled=model.main_residual_enabled,
                main_fusion_gradient_absent=not model.main_residual_enabled,
                auxiliary_loss_removed=aux_loss is None,
                microbatch='first original microbatch of checked update', groups=groups)
