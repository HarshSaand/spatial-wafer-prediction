"""Nested wafer CV for the non-gas Extra Trees setting search."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from run_baselines import load_wafer_arrays,group_name
from run_full_resolution_process_features import full_process_features
from run_region_aware_fusion import radial_regions
from run_domain_shape_hybrid import radial_descriptors
from run_sensor_family_benchmark import NON_GAS
D=Path('work/public_data');O=Path('outputs/wafer_improvement_2026-11-06');rec,xy=load_wafer_arrays(D/'Si_Oxide_etch_89_points.csv','postox_thickness');inc=np.stack([r['spatial'] for r in rec]);out=np.stack([r['target'] for r in rec]);keys=np.array([r['experiment_key'] for r in rec]);pk=[group_name(k) for k in keys]
trace,names=full_process_features(D,pk);short=[n.rsplit('_MV_',1)[-1] for n in names];ix=[i for i,n in enumerate(short) if n in NON_GAS];six=np.stack([trace[k][1][:6*len(names)].reshape(6,len(names)) for k in pk]);rad=radial_descriptors(inc,*radial_regions(xy));x=np.concatenate((inc,rad,six[:,:,ix].reshape(len(keys),-1)),axis=1)
configs=[('leaf1_f025',1,.25),('leaf1_f04',1,.4),('leaf2_f07',2,.7)]
def fit_predict(tr,te,leaf,mf,trees=400):
 m=ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=leaf,max_features=mf,random_state=42,n_jobs=-1)
 m.fit(x[tr],out[tr]-inc[tr]);return inc[te]+m.predict(x[te])
pred=np.empty_like(out);rows=[]
for fold,(train,test) in enumerate(KFold(5,shuffle=True,random_state=42).split(x),1):
 scores=[]
 for name,leaf,mf in configs:
  inner=[]
  for a,b in KFold(3,shuffle=True,random_state=100+fold).split(train):
   t=train[a];v=train[b];z=fit_predict(t,v,leaf,mf,trees=300)
   inner.extend(r2_score(u,w) for u,w in zip(out[v],z))
  scores.append(np.mean(inner))
 chosen=configs[int(np.argmax(scores))];pred[test]=fit_predict(train,test,chosen[1],chosen[2],trees=600)
 rows.append(dict(outer_fold=fold,selected=chosen[0],inner_mean_wafer_r2=float(max(scores)),outer_mean_wafer_r2=float(np.mean([r2_score(u,w) for u,w in zip(out[test],pred[test])]))));print(rows[-1],flush=True)
r2=np.array([r2_score(u,w) for u,w in zip(out,pred)]);pd.DataFrame(rows).to_csv(O/'nested_folds.csv',index=False);pd.DataFrame([dict(mean_wafer_r2=r2.mean(),rmse_um=np.sqrt(np.mean((out-pred)**2)),negative_wafers=(r2<0).sum())]).to_csv(O/'nested_summary.csv',index=False);np.savez_compressed(O/'nested_oof.npz',keys=keys,actual=out,predicted=pred)
print('Nested:',r2.mean(),np.sqrt(np.mean((out-pred)**2)),flush=True)
