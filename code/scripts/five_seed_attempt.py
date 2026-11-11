"""Survives scheduler loss and persists child exit status for inspected recovery."""
import argparse
import os
import re
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tracehr_recovery_queue import atomic,read,process_identity


def run_spec(spec):
    # Validate every launch field before creating a child. A metadata error must
    # never leave an experiment running without its owner/exit recorder.
    required=('command','cwd','receipt','job_id','source_manifest_sha256')
    missing=[key for key in required if key not in spec]
    if missing:raise ValueError('Missing launch fields: '+', '.join(missing))
    command=spec['command'];cwd=spec['cwd'];job_id=spec['job_id'];digest=spec['source_manifest_sha256']
    if not isinstance(command,list) or not command or not all(isinstance(x,str) and x for x in command):
        raise ValueError('command must be a nonempty list of strings')
    if not isinstance(cwd,str) or not Path(cwd).is_dir():raise ValueError('cwd must be an existing directory')
    if not isinstance(job_id,str) or not job_id:raise ValueError('job_id must be a nonempty string')
    if not isinstance(digest,str) or re.fullmatch(r'[0-9a-fA-F]{64}',digest) is None:
        raise ValueError('source_manifest_sha256 must be a SHA256 digest')
    if not isinstance(spec['receipt'],str) or not Path(spec['receipt']).is_absolute():
        raise ValueError('receipt must be an absolute path')
    receipt=Path(spec['receipt']);ownerfile=receipt.with_name(receipt.stem+'_owner.json')
    if receipt.exists() or ownerfile.exists():raise FileExistsError('Existing owner/exit record: inspect before retry')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    started=time.time();child=subprocess.Popen(command,cwd=cwd,stdin=subprocess.DEVNULL)
    owner=dict(job_id=job_id,wrapper_pid=os.getpid(),child_pid=child.pid,child_identity=process_identity(child.pid),
               source_manifest_sha256=digest,started_at=started)
    atomic(ownerfile,owner)
    rc=child.wait();atomic(receipt,dict(**owner,returncode=rc,finished_at=time.time()))
    return rc


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    sys.exit(run_spec(read(a.spec)))
