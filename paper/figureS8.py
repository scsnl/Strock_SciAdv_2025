# General import
import os, sys, argparse
from tqdm import tqdm

# Data import
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, ttest_1samp
from utils.data import behavior_acc, matching_ranges
from utils.prompt import print_title
from nn_modeling.dataset.arithmetic import ArithmeticDataset, no_transform
from functools import reduce

# Plot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, rgb2hex
from utils.plots import get_figsize, letter, plot_value_by_iteration, plot_bar_by_gain, plot_by_layer, multiplot_by_layer, violinplot_by_gain, plot, imshow_by_layer

def plot_stimuli(f, gs, img, op):
    ax = f.add_subplot(gs)
    ax.imshow(np.rollaxis(img, 0, 3))
    ax.set_ylabel(f'{op} = {eval(op):d}')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def plot_title(f, gs, title):
    ax = f.add_subplot(gs)
    ax.set_title(title, weight = 'bold', size = 20)
    ax.axis('off')
    return ax

def main(args):

    seed = 5
    np.random.seed(seed)

    # -------------------------------
    # Data parameters
    # -------------------------------

    n_max = 18
    n_sample_train = 50 # number of samples used in training per class/condition
    n_sample_test = 50 # number of samples used in test per class/condition
    sample = np.arange(n_sample_train, n_sample_train+n_sample_test) # id of sample used
    result_shown = np.arange(1+n_max)[::2]
    if args.dataset == 'h':
        tasks = [f'addsub_{n_max}']
    elif args.dataset == 'f':
        tasks =  [f'addsub_{n_max}_font']
    elif args.dataset == 'h+f' or args.dataset == 'f+h':
        tasks = [f'addsub_{n_max}{s}' for s in ['', '_font']]
    task =  '+'.join(tasks)

    # -------------------------------
    # Figure parameters
    # -----------------------------

    sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.get_cmap("coolwarm"))
    c_random = '0.7'
    c_td = sm.to_rgba(0)
    c_md = sm.to_rgba(1)

    zoom = 10
    groups = ['TD', 'MD']
    ws = np.array([1,1])
    wspace = 0.5
    hs = 0.25*np.array([1]*len(result_shown))
    hspace = 1/2**4

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, top = 0.5)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path containing the dataset
    dataset_path = [f'{os.environ.get("DATA_PATH")}/{task}/stimuli' for task in tasks]
    # Path where figure are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper/addsub_{n_max}'
    os.makedirs(f'{figure_path}/png', exist_ok=True)
    os.makedirs(f'{figure_path}/pdf', exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    # only the selected sample are used
    pattern_sample = f'\\d*_\\d*(\\+|-)\\d*_({reduce(lambda a,b: f"{a}|{b}", sample)})'
    # simple global pattern containing files
    path = [f'{p}/*.png' for p in dataset_path]
    # regex matching only the files that should be used for training
    include = [f'{p}/{pattern_sample}.png' for p in dataset_path]

    dataset = ArithmeticDataset(path = path, summary = f'{os.environ.get("DATA_PATH")}/{task}/stimuli/test_notransform.pkl', include = include, transform = no_transform)

    idx = np.random.permutation(np.arange(len(dataset)))
    idx = np.arange(len(dataset))
    img, label, op = dataset[idx]
    img = img.numpy()
    op = dataset.all_op[op]

    # -------------------------------
    # Display
    # -------------------------------
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))

    for i, (operator, name, threshold) in enumerate(zip(['+', '-'], ['Addition', 'Subtraction'], [4.3, 1.5])):
        for j, result in enumerate(result_shown):
            l = np.where((label == result) & (np.char.find(op,operator)>=0))[0]
            k = l[np.random.randint(len(l))]
            ax = letter('',plot_stimuli)(f, gs[j,i], img[k], op[k])
        ax = letter('',plot_title)(f, gs[:,i], name)

    f.savefig(f'{figure_path}/png/figureS8.png', dpi = 1200)
    f.savefig(f'{figure_path}/pdf/figureS8.pdf', dpi = 1200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure S8 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'f', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)
