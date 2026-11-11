"""Expose only the six authorized identities to the unchanged noA entry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_tracehr_not_stability_functional as entry
from scripts.tracehr_stage2_common import NOA_MATRIX
entry.MATRIX=NOA_MATRIX
if __name__=='__main__':entry.main()
