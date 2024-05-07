import os
import numpy as np
from tqdm import tqdm
from nn_modeling.dataset.arithmetic import *
from itertools import product
import pandas as pd
import sys

seed = 0
np.random.seed(seed)

# -------------------------------
# General parameters
# -------------------------------

n_max = 18 # maximal result of operation
ns = np.arange(n_max+1) # number considered in the operations
file = {s:f'{os.environ.get("OAK")}/projects/ehk/dnn-modeling-v2/raw/TD_MD_AddSub_{s}.csv' for s in ['Addition', 'Subtraction']}
proc = {'Addition': 'EquaProc', 'Subtraction': 'SubProc'}
csv_id = {
    'Addition': ['AddStim.ACC', 'AddStim.RT'],
    'Subtraction': ['SubStim.ACC', 'SubStim.RT']
}
unified_id = ['ACC', 'RT']
# -------------------------------
# Operation parameters
# -------------------------------

fmt = int_fmt(2)
var = Var(fmt)
op = [var+var, var-var]
op = [(o(*args),o.eval(*args)) for o in op for args in product(ns, repeat = o.n_args) if 0<=o.eval(*args)<=n_max]

# -------------------------------
# Paths
# -------------------------------

data_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/behavior'
os.makedirs(f'{data_path}', exist_ok=True)

# -------------------------------
# Generation loop
# -------------------------------

data = {}

for k,f in file.items():
    df = pd.read_csv(f)
    df = df[df["Procedure"]==proc[k]][['Subject', 'Stimulus'] + csv_id[k]]
    df["Stimulus"] = df["Stimulus"].map(lambda x: x.replace(" ", "").split("=")[0])
    df.rename(columns={k:v for (k,v) in zip(csv_id[k], unified_id)}, inplace = True)
    subject = df["Subject"].unique()
    for s in subject:
        if s in data:
            data[s].append(df[df["Subject"]==s])
        else:
            data[s] = [df[df["Subject"]==s]]

print('Subject: ACC, Corrected ACC, Response rate')
for k in data:
    data[k] = pd.concat(data[k]).drop(columns=['Subject'])
    acc = data[k]['ACC'].sum()/len(data[k])
    rrate = (data[k]["RT"]>0).sum()/len(data[k])
    acc2 = data[k]['ACC'][data[k]["RT"]>0].sum()/(rrate*len(data[k]))
    print(f'{k}: {acc:.2f}, {acc2:.2f}, {rrate:.2f}')
    data[k].to_csv(f'{data_path}/{k}.csv')