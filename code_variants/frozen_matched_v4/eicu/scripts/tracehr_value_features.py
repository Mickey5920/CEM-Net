"""Exact duration integrals of a known-result step path, not physiological imputation."""
import numpy as np
TAUS=np.asarray([1.,6.,24.])

def basis(z):
    z=np.asarray(z,dtype=np.float64)
    return np.stack([np.ones_like(z),z,z*z,np.maximum(z-1,0),np.maximum(-z-1,0)],-1)

def duration_summary(times,values,window=24.):
    t=np.asarray(times,dtype=np.float64);v=np.asarray(values,dtype=np.float64)
    assert t.ndim==1 and t.shape==v.shape and np.all(np.diff(t)>0)
    assert np.all((t>=0)&(t<=window)) and np.isfinite(v).all()
    if not len(t):return np.zeros(15,dtype=np.float32)
    end=np.r_[t[1:],window]
    mass=TAUS[None,:]*(np.exp(-(window-end[:,None])/TAUS)-np.exp(-(window-t[:,None])/TAUS))
    normal=TAUS*(1-np.exp(-window/TAUS))
    return ((mass.T@basis(v))/normal[:,None]).astype(np.float32).ravel()

def simple_summary(times,values,samples,window=24.):
    t=np.asarray(times);v=np.asarray(values);s=np.asarray(samples)
    if not len(t):return np.zeros(11,dtype=np.float32)
    dt=t-t.mean();den=(dt*dt).sum()
    slope=float(np.dot(dt,v-v.mean())/den)*window if den>1e-8 else 0.
    return np.asarray([1.,np.log1p(len(t)),t[0]/window,(window-t[-1])/window,
        (window-s[-1])/window,v.mean(),v.min(),v.max(),v[-1],v[-1]-v[0],slope],dtype=np.float32)
