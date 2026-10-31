"""Portable inference for the frozen Direct KAN release; no training imports."""
from pathlib import Path
import json
import numpy as np
from .formula_numpy import price,read

class FrozenPricer:
    """A specific historical fold and seed, with Vbase as an input only.

    A fold must be selected explicitly: these are walk-forward models, not a
    final model refitted on all history. Prices are KRW per 10,000 face value.
    """
    def __init__(self,model_dir,*,fold,seed=0,compressed=False):
        self.model_dir=Path(model_dir).resolve()
        manifest=json.loads((self.model_dir/'model_manifest.json').read_text(encoding='utf-8'))
        if fold not in range(4) or seed not in range(5):raise ValueError('fold=0..3; seed=0..4')
        if compressed and seed!=0:raise ValueError('Compressed formulas exist for seed0 only')
        self.fold=int(fold);self.seed=int(seed);self.compressed=bool(compressed)
        name='Compressed' if compressed else 'SelectedKAN'
        self.formula_path=self.model_dir/'formulas'/f'{name}_s{seed}_f{fold}.json'
        self.formula=read(self.formula_path)
        if self.formula.get('external_offset')!=0 or not self.formula['target'].startswith('Direct MC price'):
            raise ValueError('Expected a Direct MC-price formula with zero external offset')
        self.release_id=manifest['release_id']

    def predict(self,curve10,vc10,contract52):
        """Explicit Clark Vbase must be contract52[:,51], in face-value-1 units."""
        return price(curve10,vc10,contract52,self.formula)

    def predict_raw(self,curve10,vc10,contract51,nobs,nstar,*,device='cpu'):
        """Recompute the frozen Clark proxy and evaluate the exact NumPy model."""
        from ._frozen.vbase import vbase_els
        u,v,c=[np.atleast_2d(np.asarray(x,dtype='float32')) for x in [curve10,vc10,contract51]]
        if u.shape!=(len(c),10) or v.shape!=(len(c),10) or c.shape[1]!=51:
            raise ValueError('Expected aligned curve10, VC10, contract51')
        if not all(np.isfinite(x).all() for x in [u,v,c]):raise ValueError('Inputs must be finite')
        obs=np.atleast_1d(nobs);stars=np.atleast_1d(nstar)
        if len(obs)!=len(c) or len(stars)!=len(c):raise ValueError('nobs and nstar need one entry per row')
        if not (np.isin(stars,[1,2,3]).all() and ((obs>=1)&(obs<=12)&(obs==obs.astype(int))).all()):
            raise ValueError('nstar must be 1,2,3 and nobs an integer from 1 to 12')
        vb=vbase_els(c,v,u,obs,dev=device,mode='clark',lizard=False,nstar=stars)
        return dict(price_KRW=self.predict(u,v,np.c_[c,vb].astype('float32')),
                    Vbase_input_KRW=np.asarray(vb,dtype=float)*10000,external_Vbase_offset=0)

def predict_ensemble(model_dir,*,fold,curve10,vc10,contract52):
    """Five-seed mean for a single explicit historical fold; not OOS seed-mean metrics."""
    values=np.stack([FrozenPricer(model_dir,fold=fold,seed=s).predict(curve10,vc10,contract52) for s in range(5)])
    return dict(mean_KRW=values.mean(0),seed_predictions_KRW=values)
