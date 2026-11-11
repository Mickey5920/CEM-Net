"""Survives scheduler loss and persists child exit status for inspected recovery."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tracehr_recovery_queue import atomic,read,process_identity


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();spec=read(a.spec)
    receipt=Path(spec['receipt']);assert not receipt.exists()
    started=time.time();child=subprocess.Popen(spec['command'],cwd=spec['cwd'],stdin=subprocess.DEVNULL)
    owner=dict(job_id=spec['job_id'],wrapper_pid=os.getpid(),child_pid=child.pid,child_identity=process_identity(child.pid),
               source_manifest_sha256=spec['source_manifest_sha256'],started_at=started)
    atomic(receipt.with_name(receipt.stem+'_owner.json'),owner)
    rc=child.wait();atomic(receipt,dict(**owner,returncode=rc,finished_at=time.time()))
    sys.exit(rc)
