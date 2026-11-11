"""Official TabM training path with the identical raw/encoded evidence mask."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS']);sys.path[:0]=[str(deps),*[str(p) for p in deps.glob('*.whl')]]
from scripts.cem_icde72_neural import configure_tabular,tabular_model,removed_columns
from models.cem_icde72_mask import invariance
configure_tabular()
from scripts import train_cem_additional_baseline as entry
entry.make_model=tabular_model
original_initialize=entry.initialize
def initialize(a,out,data):
    model,identity=original_initialize(a,out,data)
    import torch
    x=data.batch('train',__import__('numpy').arange(16),'cuda')[0]
    audit=dict(model._evidence_audit,**invariance(model,x,removed_columns(data)))
    entry.atomic(out/'evidence_mask_audit.json',audit)
    # A deterministic diagnostic in eval mode consumes no training RNG.
    return model,identity
entry.initialize=initialize
if __name__=='__main__':entry.main()
