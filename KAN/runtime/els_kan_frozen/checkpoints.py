"""Load the frozen PyTorch checkpoint using vendored architecture code."""
from pathlib import Path
import json
import numpy as np

def load_checkpoint(model_dir,*,fold,seed=0,device='cpu'):
    import torch
    from .kan_models import network
    model_dir=Path(model_dir)
    if fold not in range(4) or seed not in range(5):raise ValueError('fold=0..3; seed=0..4')
    manifest=json.loads((model_dir/'model_manifest.json').read_text(encoding='utf-8'))
    entry=next(r for r in manifest['models'] if r['fold']==fold and r['seed']==seed)
    ck=torch.load(model_dir/entry['checkpoint'],map_location='cpu',weights_only=False)
    net=network(entry['base_architecture'],10,52,192)
    net.load_state_dict(ck['state'],strict=True)
    return ck,net.to(device).eval()

def predict_checkpoint(checkpoint,network,curve10,vc10,contract52,*,device='cpu'):
    import torch
    arrays=[]
    for x,m,s in [(curve10,'um','us'),(vc10,'vm','vs'),(contract52,'cm','cs')]:
        z=np.clip((np.atleast_2d(np.asarray(x,dtype='float32'))-checkpoint[m])/checkpoint[s],-10,10)
        arrays.append(torch.tensor(z,device=device))
    with torch.no_grad():values=network.V(*arrays).detach().cpu().numpy().astype(float)
    return 10000*(checkpoint['ym']+checkpoint['ysd']*values)
