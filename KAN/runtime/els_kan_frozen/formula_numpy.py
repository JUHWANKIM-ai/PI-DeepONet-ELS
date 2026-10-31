"""Evaluate exported Direct KAN equations with NumPy only.

contract52 must contain the SAME pointwise Vbase as feature col51. It is never
added to the output. Existing Clark computation is a separate input calculation.
"""
import json
from pathlib import Path
import numpy as np

def read(formula):return json.loads(Path(formula).read_text(encoding='utf-8')) if isinstance(formula,(str,Path)) else formula
def features(x,m):
 basis=m['basis']
 if basis=='adaptive3':
  a=np.exp(.25*np.tanh(m['log_gain']));b=.5*np.tanh(m['shift']);z=np.tanh(a*x+b);z0=np.tanh(b)
  return [x,z*z-z0*z0,z*z*z-z0*z0*z0]
 z=np.tanh(x)
 if basis=='tanh3':return [x,z*z,z*z*z]
 if basis=='tanh4':return [x,z,z*z,z*z*z]
 if basis=='cheb4':return [x,z,2*z*z,4*z*z*z-3*z,8*z**4-8*z*z]
 raise ValueError(basis)
def edge(x,m):return np.concatenate(features(x,m),-1)@np.asarray(m['weight']).T+np.asarray(m['bias'])
def kan(x,layers):
 for m in layers:x=edge(x,m)
 return x
def edge_jacobian(x,m):
 if m['basis']=='adaptive3':
  a=np.exp(.25*np.tanh(m['log_gain']));b=.5*np.tanh(m['shift']);z=np.tanh(a*x+b);dz=a*(1-z*z);d=[np.ones_like(x),2*z*dz,3*z*z*dz]
 else:
  z=np.tanh(x);dz=1-z*z
  if m['basis']=='tanh3':d=[np.ones_like(x),2*z*dz,3*z*z*dz]
  elif m['basis']=='tanh4':d=[np.ones_like(x),dz,2*z*dz,3*z*z*dz]
  else:d=[np.ones_like(x),dz,4*z*dz,(12*z*z-3)*dz,(32*z**3-16*z)*dz]
 w=np.asarray(m['weight']).reshape(len(m['bias']),len(d),m['di'])
 return np.einsum('obi,nbi->noi',w,np.stack(d,1))
def kan_jacobian(x,layers):
 j=np.broadcast_to(np.eye(x.shape[1]),(len(x),x.shape[1],x.shape[1])).copy()
 for m in layers:j=np.einsum('noi,nij->noj',edge_jacobian(x,m),j);x=edge(x,m)
 return x,j
def conv(x,m):
 windows=np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(0,0),(1,1))),3,axis=2)
 return np.einsum('bink,oik->bon',windows,np.asarray(m['weight']))+np.asarray(m['bias'])[None,:,None]
def normalized(curve,vc,contract,formula):
 f=read(formula);u,v,c=[np.atleast_2d(np.asarray(a,float)) for a in [curve,vc,contract]]
 if u.shape[1]!=10 or v.shape[1]!=10 or c.shape[1]!=52:raise ValueError('Expected curve10, VC10 and contract52 including explicit Vbase input')
 if not len(u) or len(u)!=len(v) or len(u)!=len(c) or not all(np.isfinite(a).all() for a in [u,v,c]):raise ValueError('Inputs require finite aligned rows')
 s=f['normalization'];return [np.clip((a-np.asarray(s[m]))/np.asarray(s[sd]),-10,10) for a,m,sd in [(u,'um','us'),(v,'vm','vs'),(c,'cm','cs')]]
def price(curve,vc,contract,formula,return_parts=False):
 f=read(formula);u,v,c=normalized(curve,vc,contract,f)
 h=np.maximum(conv(u[:,None,:],f['cnn'][0]),0).reshape(len(u),16,5,2).max(-1)
 h=np.maximum(conv(h,f['cnn'][1]),0).mean(-1)
 B=kan(np.c_[h,v],f['branch']);T=kan(c,f['trunk']);terms=B*T;s=f['normalization']
 p=10000*(float(s['ym'])+float(s['ysd'])*(terms.sum(-1)+float(f['b0'])))
 if return_parts:return dict(price_KRW=p,branch=B,trunk=T,term_KRW=10000*float(s['ysd'])*terms,intercept_KRW=10000*(float(s['ym'])+float(s['ysd'])*float(f['b0'])))
 return p
def contract_gradient(curve,vc,contract,formula):
 """Exact conditional KRW/raw-unit derivatives of all 52 contract inputs.

Vbase is held independent here. Recomputed-proxy scenario responses are separate.
"""
 f=read(formula);u,v,c=normalized(curve,vc,contract,f)
 h=np.maximum(conv(u[:,None,:],f['cnn'][0]),0).reshape(len(u),16,5,2).max(-1);h=np.maximum(conv(h,f['cnn'][1]),0).mean(-1)
 B=kan(np.c_[h,v],f['branch']);_,JT=kan_jacobian(c,f['trunk']);s=f['normalization']
 before=(np.atleast_2d(contract)-np.asarray(s['cm']))/np.asarray(s['cs'])
 return 10000*float(s['ysd'])*np.einsum('nr,nri->ni',B,JT)/np.asarray(s['cs'])*(abs(before)<10)
