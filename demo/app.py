"""Interactive public-data wafer surrogate explorer. Run: streamlit run demo/app.py"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_baselines import load_wafer_arrays,group_name
from run_full_resolution_process_features import full_process_features
from run_region_aware_fusion import radial_regions
from run_domain_shape_hybrid import radial_descriptors
from run_sensor_family_benchmark import NON_GAS,PRESSURE,RF_BIAS,THERMAL_BACKSIDE
DATA=ROOT/'work/public_data'
st.set_page_config(page_title='Wafer thickness surrogate',layout='wide')
@st.cache_resource(show_spinner='Loading public traces and fitting the demonstration model…')
def load_model():
 records,coords=load_wafer_arrays(DATA/'Si_Oxide_etch_89_points.csv','postox_thickness')
 inc=np.stack([r['spatial'] for r in records]);out=np.stack([r['target'] for r in records]);keys=np.array([r['experiment_key'] for r in records]);lots=np.array([r['lot'] for r in records]);pk=[group_name(k) for k in keys]
 trace,names=full_process_features(DATA,pk);short=[n.rsplit('_MV_',1)[-1] for n in names];chosen=[i for i,n in enumerate(short) if n in NON_GAS]
 six=np.stack([trace[k][1][:6*len(names)].reshape(6,len(names)) for k in pk]);rad=radial_descriptors(inc,*radial_regions(coords))
 sensor=six[:,:,chosen].reshape(len(keys),-1);x=np.concatenate((inc,rad,sensor),axis=1)
 model=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,max_features=.4,random_state=42,n_jobs=-1).fit(x,out-inc)
 assert x.shape==(88,242)
 return dict(model=model,coords=coords,inc=inc,out=out,keys=keys,lots=lots,x=x,sensor=sensor,short=[short[i] for i in chosen],rad=rad)
try:d=load_model()
except Exception as e:
 st.error(f'Could not load public data: {e}');st.stop()
st.title('Wafer thickness surrogate explorer')
st.caption('Public BOSCH plasma-etch data · 88 wafers · outgoing oxide thickness at 88 positions')
st.info('The scenario control changes model input summaries to illustrate sensitivity. It does not simulate plasma physics or predict a validated process intervention. The model shown here is fitted to all 88 wafers for exploration; use the out-of-fold benchmark for accuracy.')
left,right=st.columns([1,3])
with left:
 labels=[f'Lot {lot} · wafer {key}' for key,lot in zip(d['keys'],d['lots'])]
 index=st.selectbox('Select wafer',range(len(labels)),format_func=lambda i:labels[i])
 family=st.selectbox('Process-signal group to vary',['None','Pressure','RF / bias','Thermal / backside'])
 shift=st.slider('Shift in across-wafer standard deviations',-2.0,2.0,0.0,.25,disabled=family=='None')
 st.markdown('**Inputs**')
 st.write('2D: incoming thickness at 88 positions. 1D: six trace summaries for each of 24 non-gas sensor channels (mean, variation, minimum, maximum, 10th and 90th percentile).')
 st.write('**Output**: outgoing thickness at the same positions. The model predicts thickness change, then adds the incoming thickness.')
 st.write('**Exploratory wafer-held-out benchmark:** mean wafer R² 0.671; RMSE 0.03467 µm. This model display is fitted on all wafers and is not itself an accuracy test.')
base=d['x'][index].copy();scenario=base.copy();groups={'Pressure':PRESSURE,'RF / bias':RF_BIAS,'Thermal / backside':THERMAL_BACKSIDE};start=88+d['rad'].shape[1]
if family!='None':
 ids=[j for j,name in enumerate(d['short']) if name in groups[family]]
 for block in range(6):
  for j in ids:
   col=block*len(d['short'])+j
   scenario[start+col]+=shift*np.std(d['sensor'][:,col])
baseline=d['inc'][index]+d['model'].predict(base[None,:])[0];changed=d['inc'][index]+d['model'].predict(scenario[None,:])[0];actual=d['out'][index]
xy=d['coords']/1000.0
with right:
 fig,axes=plt.subplots(1,4,figsize=(14,3.8),layout='constrained');lo=min(actual.min(),baseline.min(),changed.min());hi=max(actual.max(),baseline.max(),changed.max())
 for ax,z,title in zip(axes[:3],[actual,baseline,changed],['Dataset outgoing','Baseline prediction','Scenario prediction']):
  im=ax.scatter(xy[:,0],xy[:,1],c=z,cmap='viridis',vmin=lo,vmax=hi,s=55);ax.set_title(title)
 delta=changed-baseline;lim=max(abs(delta).max(),1e-5);im2=axes[3].scatter(xy[:,0],xy[:,1],c=delta,cmap='coolwarm',vmin=-lim,vmax=lim,s=55);axes[3].set_title('Scenario − baseline')
 for ax in axes:ax.set_aspect('equal');ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
 fig.colorbar(im,ax=axes[:3],label='Oxide thickness (µm)',shrink=.75);fig.colorbar(im2,ax=axes[3],label='Change (µm)',shrink=.75)
 st.pyplot(fig);plt.close(fig)
 st.write(f'Baseline mean: {baseline.mean():.4f} µm · Scenario mean: {changed.mean():.4f} µm · Mean difference: {(changed-baseline).mean():+.4f} µm')
 st.caption('Dots are the dataset positions. No smooth surface or unmeasured locations are invented. Dataset outgoing values may include author-filled readings.')
st.download_button('Download this wafer’s positions and predictions',pd.DataFrame({'x_mm':xy[:,0],'y_mm':xy[:,1],'incoming_um':d['inc'][index],'dataset_outgoing_um':actual,'baseline_predicted_um':baseline,'scenario_predicted_um':changed}).to_csv(index=False),file_name='wafer_surrogate_view.csv',mime='text/csv')
