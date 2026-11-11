"""Route fixed existing model families into the seed-extension run."""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
deps = Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path[:0] = [str(deps), *[str(p) for p in deps.glob('*.whl')]]
from scripts import cem_five_seed_common as common
common.configure()
variant = sys.argv[sys.argv.index('--variant')+1]
if variant == 'CEM':
    from scripts import five_seed_cem_entry as entry
elif variant == 'LightGBM_full':
    from scripts import five_seed_lgbm_entry as entry
else:
    import torch
    from scripts import five_seed_tabular_entry as entry
    entry.make_model = common.make_tabular
    original_initialize = entry.initialize
    def initialize(a, out, data):
        model, identity = original_initialize(a, out, data)
        if a.variant == 'TabM_mini':
            torch.random.set_rng_state(model._initial_torch_rng)
        return model, identity
    entry.initialize = initialize
if __name__ == '__main__': entry.main()
