import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import pandas as pd

def main(args):
    pl.seed_everything(0)
    n_max = 18
    behavior_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/behavior'
    patients = pd.read_csv(f'{behavior_path}/TD_MD.csv')
    acc = {'TD':[], 'MD':[]}

    for p in range(patients.shape[0]):
        patient_id = patients["PID"][p]
        patient_group = patients["Group"][p]
        print(f'Kid: {patient_group} {patient_id}')        
        behavior = pd.read_csv(f'{behavior_path}/{patient_id}.csv')
        acc[patient_group].append(np.mean(behavior["ACC"]))
        print(acc[patient_group][-1])
    print('MEAN:', {k:np.mean(a) for k,a in acc.items()})
    print('MEDIAN:', {k:np.median(a) for k,a in acc.items()})
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute behavioral distance')
    args = parser.parse_args()
    main(args)
