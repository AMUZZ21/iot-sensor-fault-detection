import os, sys, argparse, joblib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from data_loader import load_all_batches, make_synthetic_demo_dataset, FEATURE_NAMES
from fault_injection import inject_faults
from preprocessing import clean, batch_train_test_split, scale_features
from feature_engineering import add_engineered_features, engineered_feature_names
from models import get_all_models

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW=os.path.join(ROOT,'data','raw')
OUT=os.path.join(ROOT,'models')

def main():
 p=argparse.ArgumentParser(); p.add_argument('--fault-fraction',type=float,default=.15); p.add_argument('--synthetic',action='store_true'); p.add_argument('--estimators',type=int,default=300); args=p.parse_args()
 os.makedirs(OUT,exist_ok=True)
 df=make_synthetic_demo_dataset(6000,10,42) if args.synthetic else load_all_batches(RAW)
 df=clean(df,FEATURE_NAMES); batches=sorted(df.batch.unique()); cut=max(1,int(len(batches)*.7)); trb=batches[:cut]; teb=batches[cut:]
 tr,te=batch_train_test_split(df,trb,teb)
 tr=inject_faults(tr,FEATURE_NAMES,args.fault_fraction,42); te=inject_faults(te,FEATURE_NAMES,args.fault_fraction,43)
 trf=add_engineered_features(tr,FEATURE_NAMES); tef=add_engineered_features(te,FEATURE_NAMES)
 cols=FEATURE_NAMES+engineered_feature_names(FEATURE_NAMES)
 trs,tes,scaler=scale_features(trf,tef,cols)
 X=trs[cols].values; y=trs.label.values; Xn=trs.loc[trs.label==0,cols].values
 models=get_all_models(y.mean()); model=models['RandomForest']; model.model.set_params(n_estimators=args.estimators)
 model.fit(X,y)
 bundle={'model':model,'scaler':scaler,'feature_columns':cols,'raw_feature_columns':FEATURE_NAMES,'fault_fraction':args.fault_fraction,'train_batches':trb,'test_batches':teb,'raw_train_mean':tr[FEATURE_NAMES].mean().values,'raw_train_std':tr[FEATURE_NAMES].std().values}
 joblib.dump(bundle,os.path.join(OUT,'RandomForest_bundle.joblib'))
 print('Saved:',os.path.join(OUT,'RandomForest_bundle.joblib')); print('Features:',len(cols)); print('Train rows:',len(tr),'Test rows:',len(te))
if __name__=='__main__': main()
