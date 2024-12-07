# General import
import os, sys, argparse
from tqdm import tqdm

# Data import
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, ttest_1samp
from utils.data import behavior_acc, matching_ranges
from utils.prompt import print_title

# Plot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from utils.plots import get_figsize, letter, plot_value_by_iteration, plot_value_by_gain, plot, plot_r_by_iteration

def main(args):

    seed = 0
    np.random.seed(seed)

    # -------------------------------
    # Data parameters
    # -------------------------------

    n_max = 18
    scales = np.linspace(1.0, 5.0, 17)
    if args.dataset == 'h':
        tasks = [f'addsub_{n_max}']
    elif args.dataset == 'f':
        tasks =  [f'addsub_{n_max}_font']
    elif args.dataset == 'h+f' or args.dataset == 'f+h':
        tasks = [f'addsub_{n_max}{s}' for s in ['', '_font']]
    task =  '+'.join(tasks)
    steps = np.arange(0, 3801, 100) if args.dataset in ['h', 'f'] else np.arange(0, 7601, 100)
    selected_steps = np.arange(0, 3801, 100)
    selected_steps_idx = np.where(selected_steps[None,:] == steps[:,None])[0]
    ext = f'_fixedbatchnorm'
    idx_roi = 1
    name_roi = 'right IPL/IPS'
    thresholds = [0.95]

    # -------------------------------
    # Figure parameters
    # -----------------------------

    sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.get_cmap("coolwarm"))
    c_random = '0.7'
    c_td = sm.to_rgba(0)
    c_md = sm.to_rgba(1)

    zoom = 10
    groups = ['TD', 'MD']
    phi = (1+np.sqrt(5))/2
    ws = phi*np.array([1,1])
    wspace = phi
    hs = np.array([1,1])
    hspace = 1

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # Path where accuracy are saved
    accuracy_path = f'{os.environ.get("DATA_PATH")}/{task}/accuracy'
    # Path where figure are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper/{task}'
    os.makedirs(f'{figure_path}/png', exist_ok=True)
    os.makedirs(f'{figure_path}/pdf', exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    # Model accuracy

    acc = np.empty((len(scales), len(steps)))
    for i, scale in enumerate(tqdm(scales, leave = False)):
        if scale%0.5 == 0:
            accuracy = np.load(f'{accuracy_path}/scaled_{scale:.1f}{ext}/steps_{steps[0]}_{steps[-1]}_accuracy.npy')
        else:
            accuracy = np.load(f'{accuracy_path}/scaled_{scale:.2f}{ext}/steps_{steps[0]}_{steps[-1]}_accuracy.npy')
        idx = np.where(accuracy["step"][:,None] == steps[None, :])[0]
        acc[i] = accuracy["accuracy"][idx]

    # Model iteration to reach threshold accuracy

    step_reach_threshold = {}
    for t in thresholds:
        step_reach_threshold[t] = np.zeros(len(scales), dtype = steps.dtype)
        for i, scale in enumerate(tqdm(scales, leave = False)):
            idx = np.where(acc[i]>= t)[0]
            step_reach_threshold[t][i] = steps[idx[0]]\

    # Correlation accuracy and gain across iteration
    
    rs_acc = np.empty(len(selected_steps))
    ps_acc = np.empty(len(selected_steps))
    for i, j in enumerate(selected_steps_idx):
        try:
            rs_acc[i], ps_acc[i] = pearsonr(acc[:,j], scales)
        except:
            rs_acc[i], ps_acc[i] = np.nan, np.nan

    # -------------------------------
    # Stats
    # -------------------------------
    
    print_title('Debug')
    print(f'acc shape: {acc.shape}')
    print(f'figsize: {figsize}')
    print_title('Stats')
    for i, scale in enumerate(scales):
        print(f'G = {scale:.1f}: reaching 0.95 at iteration {step_reach_threshold[0.95][i]:d}')
    print(f'Best accuracy iteration {steps[0]:d}: {np.max(acc[:,0]):.3f}')
    print(f'Worse accuracy iteration {selected_steps[-1]:d}: {np.min(acc[:,selected_steps_idx[-1]]):.3f}')
    print(f'Worse accuracy iteration {steps[-1]:d}: {np.min(acc[:,-1]):.3f}')
    print(f'Correlation accuracy vs G iteration {steps[2]:d}: (r = {rs_acc[2]:.3f}, p = {ps_acc[2]:.2e})')

    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    ax_A = letter('A',plot_value_by_iteration)(f, gs[0,0], acc[:,selected_steps_idx], scales, selected_steps, ylabel = 'accuracy', clabel = 'gain $G$', ylim = (-0.1, 1.1), cticks = np.arange(1,6), threshold = 0.95, chance = 1/19)
    ax_B = letter('B',plot)(f, gs[0,1], scales, step_reach_threshold[0.95], xlabel = 'gain $G$', ylabel = f'required iteration', xlim = (0.5, 5.5), ylim = (0, 5000), loc = 'lower right', correlate = True)
    print(f'Correlation gain $G$ vs accuracy: (r = {rs_acc[1]:.3f}, p = {ps_acc[1]:.2e})')
    ax_C = letter('C',plot_value_by_gain)(f, gs[1,0], acc[:,selected_steps_idx], scales, selected_steps, ylabel = 'accuracy', clabel = 'iteration', ylim = (-0.1, 1.1))
    ax_D = letter('D',plot_r_by_iteration)(f, gs[1,1], rs_acc, ps_acc, selected_steps, legend = True)
    f.savefig(f'{figure_path}/png/figure2.png', dpi = 1200)
    f.savefig(f'{figure_path}/pdf/figure2.pdf', dpi = 1200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 2 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)
