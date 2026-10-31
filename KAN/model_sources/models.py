"""Direct-target KAN variants. Keep the existing curve CNN and input blocks."""
import math
import torch
from torch import nn
from module.networks import CurveOperatorV2

SPECS={
 'BaselineKAN':dict(basis='tanh3',branch=[42,128,192,192],trunk=[52,192,192,192]),
 'AdaptiveKAN192':dict(basis='adaptive3',branch=[42,128,192,192],trunk=[52,192,192,192]),
 'Tanh4KAN192':dict(basis='tanh4',branch=[42,128,192,192],trunk=[52,192,192,192]),
 'Cheb4KAN128':dict(basis='cheb4',branch=[42,128,128,128],trunk=[52,128,128,128]),
 'ShallowKAN64':dict(basis='tanh4',branch=[42,128,64],trunk=[52,128,64])}

class Edge(nn.Module):
 def __init__(self,di,do,basis):
  super().__init__();self.di=di;self.basis=basis
  nb=dict(tanh3=3,adaptive3=3,tanh4=4,cheb4=5)[basis]
  self.weight=nn.Parameter(torch.zeros(do,nb*di));self.bias=nn.Parameter(torch.zeros(do))
  nn.init.xavier_uniform_(self.weight[:,:di],gain=.5)
  with torch.no_grad():self.weight[:,di:].normal_(0,.025/math.sqrt(di))
  if basis=='adaptive3':
   self.log_gain=nn.Parameter(torch.zeros(di));self.shift=nn.Parameter(torch.zeros(di))
 def forward(self,x):
  if self.basis=='adaptive3':
   a=torch.exp(.25*torch.tanh(self.log_gain));b=.5*torch.tanh(self.shift);z=torch.tanh(a*x+b);z0=torch.tanh(b);features=[x,z*z-z0*z0,z*z*z-z0*z0*z0]
  else:
   z=x.tanh()
   if self.basis=='tanh3':features=[x,z*z,z*z*z]
   elif self.basis=='tanh4':features=[x,z,z*z,z*z*z]
   else:features=[x,z,2*z*z,4*z*z*z-3*z,8*z**4-8*z*z]
  return nn.functional.linear(torch.cat(features,-1),self.weight,self.bias)

class DirectKAN(CurveOperatorV2):
 def __init__(self,name,nvc=10,ncon=52,P=192):
  super().__init__(nvc,ncon,P);s=SPECS[name];assert nvc==10 and ncon==52
  with torch.random.fork_rng(devices=[]):
   self.branch=nn.Sequential(*[Edge(a,b,s['basis']) for a,b in zip(s['branch'][:-1],s['branch'][1:])])
   self.t=nn.Sequential(*[Edge(a,b,s['basis']) for a,b in zip(s['trunk'][:-1],s['trunk'][1:])])

def network(name,nvc=10,ncon=52,P=192):return DirectKAN(name,nvc,ncon,P)
