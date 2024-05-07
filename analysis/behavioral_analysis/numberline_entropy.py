import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import pandas as pd
from tqdm import tqdm
from matplotlib import animation
import itertools
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.transforms as transforms
from scipy.special import softmax
from scipy.stats import entropy

def plot_value_by_iteration(f, gs, value, es, steps, labely, labelc, xlim = None, ylim = None, title = '', sharey = None):
    return ax

def main(args):
    pl.seed_everything(0)
    n_max = 18
    scales = np.linspace(1.0, 5.0, 17)
    steps = np.arange(0, 3801, 100)
    activity_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/activity'
    distribution_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/distribution'
    entropy_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/entropy'
    ext = f'_{args.variant}' if len(args.variant) > 0 else ''
    os.makedirs(distribution_path, exist_ok = True)
    os.makedirs(entropy_path, exist_ok = True)

    filename = [f'{distribution_path}/n_summary_scaled{ext}.npy']
    task = np.load(f'{activity_path}/task.npz')
    label = task["label"]
    op = task["op"]
    all_op = task["all_op"]
    add_op = np.where(np.char.find(all_op,"+") >= 0)[0]
    sub_op = np.where(np.char.find(all_op,"-") >= 0)[0]
    all_op_switch = np.copy(all_op)
    all_op_switch[add_op] = np.char.replace(all_op[add_op], '+', '-')
    all_op_switch[sub_op] = np.char.replace(all_op[sub_op], '-', '+')
    label2 = np.array([eval(op) for op in all_op_switch])[op]

    if args.redo or not np.all([os.path.exists(f) for f in filename]):
        n = np.zeros((len(scales), len(steps), len(all_op), n_max+1), dtype = np.int64)
        for i, scale in enumerate(tqdm(scales, leave = False)):
            if scale%0.5 == 0:
                model_name = f'scaled_{scale:.1f}{ext}'
            else:
                model_name =  f'scaled_{scale:.2f}{ext}'
            for j, step in enumerate(tqdm(steps, leave = False)):
                decoder = np.load(f'{activity_path}/{model_name}/step{step:05}/decoder.npz')["decoder"]
                choice = np.argmax(decoder, axis = 2)
                for k,o in enumerate(all_op):
                    idx = op == k
                    c, _n = np.unique(choice[idx], return_counts = True)
                    n[i,j,k,c] = _n
        np.save(filename[0], n)
    else:
        n = np.load(filename[0])

    p = np.sum(n, axis = 2) / np.sum(np.sum(n, axis = 2), axis = 2, keepdims = True)
    h = entropy(p, axis = 2)
    np.save(f'{entropy_path}/response.npy', h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute distribution and entropy of behavioral response')
    parser.add_argument('--variant', metavar = 'V', type = str, default = 'fixedbatchnorm', help = 'variant used')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
