"""Function-preserving KAN extensions initialized from each own-fold baseline."""
import torch
import models
SPECS={
 'RefinePlain':dict(initialization='own-fold own-seed BaselineKAN',base_architecture='BaselineKAN'),
 'RefineAdaptive':dict(initialization='same initial function, add input gain/shift',base_architecture='AdaptiveKAN192'),
 'RefineTanh4':dict(initialization='same initial function, insert zero-weight tanh basis',base_architecture='Tanh4KAN192')}
def build(name,checkpoint,nvc=10,ncon=52,P=192):
 net=models.network(SPECS[name]['base_architecture'],nvc,ncon,P);old=checkpoint['state'];new=net.state_dict()
 for key,value in new.items():
  if key not in old:continue
  if value.shape==old[key].shape:value.copy_(old[key])
  else:
   assert name=='RefineTanh4' and key.endswith('.weight') and key.startswith(('branch.','t.'))
   di=old[key].shape[1]//3;assert value.shape[1]==4*di
   value.zero_();value[:,:di].copy_(old[key][:,:di]);value[:,2*di:].copy_(old[key][:,di:])
 net.load_state_dict(new);return net
