"""Use an existing local frozen package; no model parameters are bundled here."""
from pathlib import Path
import sys,argparse,json
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import numpy as np
from els_kan_frozen import FrozenPricer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);args=ap.parse_args();root=args.model_dir.resolve()
    if not (root/'examples/input.npz').is_file():ap.error('A separately supplied complete Direct_KAN_20261031_v1 folder is required; it is not included in this code-only branch.')
    with np.load(root/'examples/input.npz') as x:
        model=FrozenPricer(root,fold=3,seed=0);p=model.predict(x['curve'],x['vc'],x['contract52'])
        np.testing.assert_allclose(p,x['expected_KRW'],rtol=0,atol=1e-7)
    print(json.dumps(dict(fold=3,seed=0,price_KRW=p.tolist(),scope='Uses separately supplied local frozen model files'),indent=2))
if __name__=='__main__':main()
