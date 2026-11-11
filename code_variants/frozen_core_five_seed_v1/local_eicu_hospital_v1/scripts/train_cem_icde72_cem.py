"""Dispatch the fixed original CEM training path with one evidence mask."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS']);sys.path[:0]=[str(deps),*[str(p) for p in deps.glob('*.whl')]]
from scripts import cem_evidence_common as common
from scripts.cem_icde72_common import expected
from scripts.cem_icde72_neural import fresh_cem
common.expected=expected
common.variants=lambda:('CEM_hospital',) if expected()['stage']=='hospital' else ('CEM_basic',)
common.matrix=lambda:expected()['matrix']
common.fresh_model=fresh_cem
if expected()['stage']=='hospital':
    from scripts.cem_icde72_hospital import HospitalData,fresh_cem as hospital_cem
    common.Data=HospitalData
    common.fresh_model=hospital_cem
from scripts.train_cem_evidence_controls import main
if __name__=='__main__':main()
