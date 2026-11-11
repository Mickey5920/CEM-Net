"""Route seed extension to unchanged archived training implementations."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
deps = Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path[:0] = [str(deps), *map(str, deps.glob('*.whl'))]
from scripts import cem_core_five_seed_common as common
common.configure()
variant = sys.argv[sys.argv.index('--variant')+1]
if variant.startswith(('EMA_', 'CEM_')):
    from scripts import core_seed_cem_entry as entry
elif variant.startswith('LightGBM'):
    from scripts import core_seed_lgbm_entry as entry
elif variant == 'XGBoost_TD':
    from scripts import core_seed_xgb_entry as entry
elif variant == 'STraTS_supervised':
    from scripts import core_seed_strats_entry as route
    entry = route.worker
else:
    import torch
    from scripts import core_seed_tabular_entry as entry
    entry.make_model = common.tabular_model
    if common.expected()['stage'] in ('basic','hospital'):
        from scripts.cem_icde72_neural import tabular_protocol
        entry.protocol = tabular_protocol
    original_initialize = entry.initialize
    def initialize(a, out, data):
        model, identity = original_initialize(a, out, data)
        # Preserve each archived worker's RNG boundary. The basic-mask worker
        # explicitly reseeds after construction, including its no-PLE arm.
        if a.variant == 'TabM_mini':
            torch.random.set_rng_state(model._initial_torch_rng)
        return model, identity
    entry.initialize = initialize
if __name__ == '__main__': entry.main()
