"""P2 MiBici 2024: baselines de clasificación de presión operativa proxy.

Ejecutar desde la raíz del repositorio:
 python -m mibici.modeling.p2_baseline --tracking-uri sqlite:///mlflow.db

La etiqueta es el balance absoluto / capacidad en la hora objetivo (proxy,
NO inventario real). Solo se utilizan características conocidas antes de esa hora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score

EXPERIMENT = 'mibici-equipo15'
COLS = ['viaje_id','usuario_id','genero','anio_nacimiento','inicio','fin','origen','destino']
SPLITS = {'train':('2024-01-01','2024-09-01'), 'val':('2024-09-01','2024-11-01'), 'test':('2024-11-01','2025-01-01')}

def project_root():
    return Path(__file__).resolve().parents[2]

def sha256_file(path, chunk_bytes=1024*1024):
    digest=hashlib.sha256()
    with open(path,'rb') as h:
        for block in iter(lambda:h.read(chunk_bytes), b''):
            digest.update(block)
    return digest.hexdigest()

def git_commit(root):
    res=subprocess.run(['git','rev-parse','HEAD'],cwd=root, capture_output=True,text=True,check=False)
    return res.stdout.strip() if res.returncode == 0 else 'uncommitted'

def load_capacities(path):
    raw=pd.read_csv(path,dtype=str,encoding='utf-8-sig')
    if not {'id','dpcapacity'}.issubset(raw.columns):
        raise ValueError('Nomenclatura requiere columnas id y dpcapacity')
    x=raw[['id','dpcapacity']].copy()
    x['id']=x['id'].astype('string').str.strip()
    x['capacity']=pd.to_numeric(x['dpcapacity'], errors='coerce')
    valid=x['id'].notna() & x['id'].ne('') & x['capacity'].gt(0) & x['capacity'].eq(np.floor(x['capacity']))
    x=x.loc[valid]
    conflict=set(x.groupby('id')['capacity'].nunique().loc[lambda s:s>1].index)
    x=x.loc[~x['id'].isin(conflict)].drop_duplicates('id')
    return x.set_index('id')['capacity'].astype('int32'), {'catalog_total':len(raw),'valid_capacities':len(x),'conflicting_ids':len(conflict)}

def aggregate_flows(raw,chunk_size):
    files=sorted(raw.glob('datos_abiertos_2024_*.csv'))
    if [p.stem[-2:] for p in files] != [f'{m:02d}' for m in range(1,13)]:
        raise ValueError('Se requieren los doce CSV de enero a diciembre de 2024')
    parts=[]; manifest=[]; bad_dates=0; missing_stations=0
    for path in files:
        m=int(path.stem[-2:]); fmt='%d/%m/%Y %H:%M' if m in (9,10) else '%Y-%m-%d %H:%M:%S'
        enc='utf-8' if m==7 else 'latin1'
        n=0
        for chunk in pd.read_csv(path,dtype=str,encoding=enc,chunksize=chunk_size):
            if len(chunk.columns)!=8:
                raise ValueError(f'{path.name}: columnas inesperadas ({len(chunk.columns)})')
            chunk.columns=COLS; n+=len(chunk)
            for dt_col,st_col,value in [('inicio','origen','salidas'),('fin','destino','llegadas')]:
                times=pd.to_datetime(chunk[dt_col],format=fmt,errors='coerce')
                stations=chunk[st_col].astype('string').str.strip()
                bad_dates+=int(times.isna().sum())
                missing_stations+=int((stations.isna() | stations.eq('')).sum())
                good=times.notna() & stations.notna() & stations.ne('') & times.ge(pd.Timestamp('2024-01-01')) & times.lt(pd.Timestamp('2025-01-01'))
                if good.any():
                    tmp=pd.DataFrame({'station':stations.loc[good].astype(str).to_numpy(), 'hour':times.loc[good].dt.floor('h').to_numpy()})
                    g=tmp.groupby(['station','hour'],observed=True).size().rename(value).reset_index()
                    parts.append(g)
        manifest.append({'name':path.name,'rows':n,'sha256':sha256_file(path)})
        print(f'{path.name}: {n:,} viajes',flush=True)
    if not parts: raise ValueError('No se encontraron flujos válidos')
    combined=pd.concat(parts,ignore_index=True)
    counts=combined.groupby(['station','hour'],as_index=False)[['salidas','llegadas']].sum().fillna(0)
    counts[['salidas','llegadas']]=counts[['salidas','llegadas']].astype('int32')
    return counts, {'files':manifest,'invalid_dates':bad_dates,'missing_stations':missing_stations}

def full_panel(observed,stations,start,end,capacity):
    hours=pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(hours=1),freq='h')
    ix=pd.MultiIndex.from_product([stations,hours],names=['station','hour'])
    observed_idx=observed.set_index(['station','hour'])[['salidas','llegadas']]
    f=observed_idx.reindex(ix,fill_value=0).reset_index()
    f[['salidas','llegadas']]=f[['salidas','llegadas']].astype('int32')
    f['capacity']=f['station'].map(capacity).astype('int32')
    f['balance']=f['llegadas']-f['salidas']
    f['pressure']=f['balance'].abs()/f['capacity']
    f['weekday']=f['hour'].dt.dayofweek.astype('int8')
    f['hour_of_day']=f['hour'].dt.hour.astype('int8')
    return f

def assign_label(frame,threshold):
    # El balance de la hora futura se utiliza SOLO para definir la etiqueta.
    frame['y']=(frame['pressure']>=threshold).astype('int8')
    return frame

def scores(y,pred,prob):
    out={'precision':float(precision_score(y,pred,zero_division=0)),
         'recall':float(recall_score(y,pred,zero_division=0)),
         'f1':float(f1_score(y,pred,zero_division=0)),
         'positive_rate':float(np.mean(y)),
         'predicted_positive_rate':float(np.mean(pred))}
    out['average_precision']=float(average_precision_score(y,prob)) if np.any(y) else 0.0
    return out

def historic_prob(train,frame,group_mean,station_mean,global_mean):
    idx=pd.MultiIndex.from_frame(frame[['station','weekday','hour_of_day']])
    p=group_mean.reindex(idx).to_numpy(dtype='float64')
    backup=frame['station'].map(station_mean).fillna(global_mean).to_numpy(dtype='float64')
    return np.where(np.isfinite(p),p,backup)

def plot_cm(y,pred,path,title):
    cm=confusion_matrix(y,pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,4))
    im=ax.imshow(cm,cmap='Blues')
    for (r,c),value in np.ndenumerate(cm):ax.text(c,r,f'{value:,}',ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicción',ylabel='Etiqueta proxy',title=title)
    fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-uri',default=None,help='Ej. sqlite:///mlflow.db')
    parser.add_argument('--threshold',type=float,default=.25,help='Umbral de presión proxy (default 0.25)')
    parser.add_argument('--chunk-size',type=int,default=100000)
    args=parser.parse_args()
    if not 0<args.threshold<=1: parser.error('--threshold debe estar entre 0 (exclusivo) y 1')
    root=project_root(); raw=root/'data'/'raw'
    capacity,cap_meta=load_capacities(raw/'nomenclatura_2026_09.csv')
    flows,manifest=aggregate_flows(raw,args.chunk_size)
    # No utilizar estaciones que aparecen por primera vez en validación o prueba.
    train_stations=pd.Index(sorted(set(flows.loc[flows['hour']<pd.Timestamp('2024-09-01'),'station']) & set(capacity.index)))
    observed=set(flows['station']); excluded=sorted(observed-set(train_stations))
    print(f'Estaciones entrenamiento con capacidad: {len(train_stations)}; fuera del universo: {len(excluded)}')
    if len(train_stations)==0:raise ValueError('No hay estaciones válidas en entrenamiento')
    split={name:assign_label(full_panel(flows,train_stations,a,b,capacity),args.threshold) for name,(a,b) in SPLITS.items()}
    for key,f in split.items():print(f'{key}: {len(f):,} estación-horas, positivos {int(f.y.sum()):,} ({f.y.mean():.3%})')
    train,val,test=(split[x] for x in ['train','val','test'])
    global_prob=float(train.y.mean())
    if global_prob==0:raise ValueError('No hay clase positiva en entrenamiento; reconsidera umbral')
    group_mean=train.groupby(['station','weekday','hour_of_day'],observed=True)['y'].mean()
    station_mean=train.groupby('station',observed=True)['y'].mean()
    # El umbral de probabilidad para el baseline histórico se determina SOLO con validación.
    val_prob=historic_prob(train,val,group_mean,station_mean,global_prob)
    candidates=np.round(np.linspace(.01,.50,50),4)
    val_f1=[f1_score(val.y,(val_prob>=t).astype('int8'),zero_division=0) for t in candidates]
    selected=float(candidates[int(np.argmax(val_f1))])
    print('Umbral probabilidad histórica seleccionado en validación:',selected)
    if args.tracking_uri:mlflow.set_tracking_uri(args.tracking_uri)
    elif not __import__('os').environ.get('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    manifest['nomenclature']={'name':'nomenclatura_2026_09.csv','sha256':sha256_file(raw/'nomenclatura_2026_09.csv'),**cap_meta}
    manifest['periods']=SPLITS
    manifest['excluded_stations_count']=len(excluded)
    with tempfile.TemporaryDirectory(prefix='mibici_p2_') as tmp:
        temp=Path(tmp)
        (temp/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        for mode in ['dummy_most_frequent','historical_station_weekday_hour']:
            run_name=f'P2_baseline_{mode}'
            all_scores={}
            for split_name, f in split.items():
                p=(np.zeros(len(f),dtype='float64') if mode=='dummy_most_frequent' else historic_prob(train,f,group_mean,station_mean,global_prob))
                pred=(np.zeros(len(f),dtype='int8') if mode=='dummy_most_frequent' else (p>=selected).astype('int8'))
                all_scores[split_name]=scores(f.y.to_numpy(),pred,p)
                if split_name=='test': plot_cm(f.y,pred,temp/'test_confusion_matrix.png',run_name)
            (temp/'metrics.json').write_text(json.dumps(all_scores,indent=2),encoding='utf-8')
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({'model':mode,'proxy_threshold':args.threshold,'probability_threshold':selected if mode!='dummy_most_frequent' else 'not_applicable',
                    'threshold_selection':'best_validation_f1_grid_0.01_0.50' if mode!='dummy_most_frequent' else 'most_frequent_train',
                    'train_period':'2024-01_to_2024-08','val_period':'2024-09_to_2024-10','test_period':'2024-11_to_2024-12',
                    'train_stations':len(train_stations),'excluded_stations':len(excluded),'chunk_size':args.chunk_size,'seed':42,
                    'zero_hour_policy':'fill_zero','label':'abs(arrivals-departures)/capacity>=threshold',
                    'capacity_source':'nomenclatura_2026_09.csv'})
                mlflow.set_tags({'team':'15','question':'P2','dataset':'MiBici_2024','data_version':'sha256_manifest',
                     'target':'next_hour_proxy_pressure_risk','prediction_horizon':'next_hour','git_commit':git_commit(root),
                     'limitation':'proxy_not_observed_inventory'})
                for set_name, met in all_scores.items():
                    for metric,value in met.items():mlflow.log_metric(f'{set_name}_{metric}',value)
                mlflow.log_artifact(str(temp/'metrics.json'),'p2')
                mlflow.log_artifact(str(temp/'data_manifest.json'),'p2')
                mlflow.log_artifact(str(temp/'test_confusion_matrix.png'),'p2')
            print(run_name,'test:',all_scores['test'],flush=True)
    print('Listo: runs P2 registrados en',EXPERIMENT)

if __name__=='__main__':
    main()
