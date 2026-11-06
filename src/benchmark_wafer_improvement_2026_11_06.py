"""Exploratory matched five-fold wafer benchmark; all reported scores are out of fold."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from run_baselines import load_wafer_arrays,group_name
from run_full_resolution_process_features import full_process_features
from run_region_aware_fusion import radial_regions
from run_domain_shape_hybrid import radial_descriptors
from run_sensor_family_benchmark import NON_GAS
D=Path('work/public_data');O=Path('outputs/wafer_improvement_2026-11-06');O.mkdir(exist_ok=True)
rec,xy=load_wafer_arrays(D/'Si_Oxide_etch_89_points.csv','postox_thickness');inc=np.stack([r['spatial'] for r in rec]);out=np.stack([r['target'] for r in rec]);keys=np.array([r['experiment_key'] for r in rec]);lots=np.array([r['lot'] for r in rec]);pk=[group_name(k) for k in keys]
trace,names=full_process_features(D,pk);short=[n.rsplit('_MV_',1)[-1] for n in names];ix=[i for i,n in enumerate(short) if n in NON_GAS];six=np.stack([trace[k][1][:6*len(names)].reshape(6,len(names)) for k in pk]);rad=radial_descriptors(inc,*radial_regions(xy));x=np.concatenate((inc,rad,six[:,:,ix].reshape(len(keys),-1)),axis=1)
assert x.shape==(88,242)
configs=[('extra_leaf1_f04',ExtraTreesRegressor,n:=dict(n_estimators=600,min_samples_leaf=1,max_features=.4,random_state=42,n_jobs=-1)),('extra_leaf1_f07',ExtraTreesRegressor,dict(n_estimators=600,min_samples_leaf=1,max_features=.7,random_state=42,n_jobs=-1)),('extra_leaf2_f04',ExtraTreesRegressor,dict(n_estimators=600,min_samples_leaf=2,max_features=.4,random_state=42,n_jobs=-1)),('extra_leaf1_f025',ExtraTreesRegressor,dict(n_estimators=600,min_samples_leaf=1,max_features=.25,random_state=42,n_jobs=-1)),('extra_leaf1_f035',ExtraTreesRegressor,dict(n_estimators=600,min_samples_leaf=1,max_features=.35,random_state=42,n_jobs=-1)),('extra_leaf1_f05',ExtraTreesRegressor,dict(n_estimators=600,min_samples_leaf=1,max_features=.5,random_state=42,n_jobs=-1))]
rows=[];pred={}
for label,cls,kw in configs:
 p=np.empty_like(out)
 for tr,te in KFold(5,shuffle=True,random_state=42).split(x):
  m=cls(**kw);m.fit(x[tr],out[tr]-inc[tr]);p[te]=inc[te]+m.predict(x[te])
 scores=np.array([r2_score(a,b) for a,b in zip(out,p)]);rows.append(dict(model=label,mean_wafer_r2=scores.mean(),rmse_um=np.sqrt(np.mean((out-p)**2)),negative_wafers=(scores<0).sum()));pred[label]=p;print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(O/'summary.csv',index=False);np.savez_compressed(O/'oof.npz',keys=keys,lots=lots,actual=out,**pred)
(O/'README.txt').write_text('Exploratory model-setting sweep on the same five wafer folds. Candidate selection on these folds is not independent confirmation.\n')
