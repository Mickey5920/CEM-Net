import ctypes,os,sys,subprocess,json
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(41,1,0,0,0)==0,ctypes.get_errno()
assert libc.prctl(42,0,0,0,0)==1
assert os.environ['NUMPY_MADVISE_HUGEPAGE']=='0'
# Check inheritance before launching the original unmodified queue.
check=subprocess.check_output([sys.executable,'-B','-c',"import ctypes; print(ctypes.CDLL(None).prctl(42,0,0,0,0))"],text=True)
assert check.strip()=='1'
print(json.dumps(dict(process_local_THP_disabled=True,exec_inheritance_verified=True,numpy_hugepage_advice=False)),flush=True)
os.execv(sys.executable,[sys.executable,'-B']+sys.argv[1:])
