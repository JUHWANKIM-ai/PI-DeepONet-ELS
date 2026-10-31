"""Price issuance inputs with an exported Direct KAN equation.

The equation evaluator uses only NumPy. Computing the existing Clark Vbase from
raw inputs additionally requires this repository and PyTorch.
"""
from pathlib import Path
import sys
import numpy as np
from formula_numpy import read,price
class FormulaPricer:
 def __init__(self,formula):
  self.formula=read(formula)
  if self.formula.get('external_offset')!=0 or not self.formula.get('target','').startswith('Direct MC price'):raise ValueError('Expected an exported Direct MC-target formula')
 def predict(self,curve,vc,contract52):return price(curve,vc,contract52,self.formula)
 def predict_raw(self,curve,vc,contract51,nobs,nstar,device='cpu'):
  root=next(p for p in Path(__file__).resolve().parents if (p/'module/vbase.py').is_file())
  if str(root) not in sys.path:sys.path.insert(0,str(root))
  from module.vbase import vbase_els
  u,v,c=[np.atleast_2d(np.asarray(a,dtype='float32')) for a in [curve,vc,contract51]]
  if u.shape[1]!=10 or v.shape[1]!=10 or c.shape[1]!=51:raise ValueError('Expected curve10, VC10 and contract51')
  if len(u)!=len(v) or len(u)!=len(c):raise ValueError('Input rows must match')
  nobs=np.atleast_1d(nobs);nstar=np.atleast_1d(nstar)
  if len(nobs)!=len(u) or len(nstar)!=len(u):raise ValueError('nobs/nstar require one value per input row')
  vb=vbase_els(c,v,u,nobs,dev=device,mode='clark',lizard=False,nstar=nstar)
  return dict(price_KRW=price(u,v,np.c_[c,vb].astype('float32'),self.formula),Vbase_input_KRW=vb.astype(float)*10000,external_Vbase_offset=0)
