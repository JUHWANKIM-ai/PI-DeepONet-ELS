"""Verify network shapes with random weights; this is NOT a learned pricer."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import torch
from els_kan_frozen.kan_models import network,SPECS

def main():
    torch.manual_seed(0);torch.set_num_threads(2);rows=[]
    for name in ['AdaptiveKAN192','Tanh4KAN192','BaselineKAN']:
        model=network(name,10,52,192).eval()
        with torch.no_grad():
            out=model.V(torch.zeros(2,10),torch.zeros(2,10),torch.zeros(2,52))
        assert out.shape==(2,) and torch.isfinite(out).all()
        rows.append(dict(model=name,branch=SPECS[name]['branch'],trunk=SPECS[name]['trunk'],parameters=sum(p.numel() for p in model.parameters()),output_shape=list(out.shape)))
    print(json.dumps(dict(scope='Architecture only, random weights. No trained prices or checkpoint data included.',models=rows),indent=2))
if __name__=='__main__':main()
