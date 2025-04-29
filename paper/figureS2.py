# General import
import os, sys, argparse
from tqdm import tqdm

# Data import
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, ttest_1samp
from utils.data import behavior_acc, matching_ranges, cohen_d
from utils.prompt import print_title

# Plot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, rgb2hex
from utils.plots import get_figsize, letter, plot_value_by_iteration, plot_bar_by_gain, plot_by_layer, multiplot_by_layer, violinplot_by_gain, plot, imshow_by_layer, violinplot

import re

split = np.vectorize((lambda s: np.array([int(_s) for _s in re.split('\+|-', s)])), signature='()->(2)')

def same_operand_idx(all_op):
    operands = split(all_op)
    idx = np.sum(operands[None] != operands[:,None], axis = -1) == 0
    np.fill_diagonal(idx, False)
    return idx

def left_same_operand_idx(all_op):
    operands = split(all_op)
    idx = (operands[None,:,0] == operands[:,None,0]) & (operands[None,:,1] != operands[:,None,1])
    return idx

def right_same_operand_idx(all_op):
    operands = split(all_op)
    idx = (operands[None,:,1] == operands[:,None,1]) & (operands[None,:,0] != operands[:,None,0])
    return idx

def different_operand_idx(all_op):
    operands = split(all_op)
    idx = np.sum(operands[None] == operands[:, None], axis = -1) == 0
    return idx

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
    best_step = 800
    best_step_idx = np.where(steps == best_step)[0][0]

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
    ws = phi*np.array([5.5,1])
    wspace = 1.5*phi
    hs = np.array([1,1,1,1])
    hspace = phi

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 1.5, right = 3)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # Path where model activity are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/{task}/activity'
    # Path where representational similarity are saved
    rs_path = f'{os.environ.get("DATA_PATH")}/{task}/rsa_test'
    # Path where figure are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper/{task}'
    os.makedirs(f'{figure_path}/png', exist_ok=True)
    os.makedirs(f'{figure_path}/pdf', exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    # Model add-sub similarity

    task = np.load(f'{activity_path}/task.npz')
    rs = np.load(f'{rs_path}/rsa_correlation_scaled{ext}.npy')
    all_op = task["all_op"]
    same_operand_rs = np.mean(rs[:,:,:,same_operand_idx(all_op)], axis = 3).T
    left_same_operand_rs = np.mean(rs[:,:,:,left_same_operand_idx(all_op)], axis = 3).T
    right_same_operand_rs = np.mean(rs[:,:,:,right_same_operand_idx(all_op)], axis = 3).T
    different_operand_rs = np.mean(rs[:,:,:,different_operand_idx(all_op)], axis = 3).T

    # -------------------------------
    # Stats
    # -------------------------------
    
    print_title('Debug')
    print(f'rs shape: {rs.shape}')
    print(f'figsize: {figsize}')

    cmap = 'viridis'
    norm  = Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    res = [eval(s) for s in all_op]
    idx_plus = np.char.find(all_op,"+")
    idx_minus = np.char.find(all_op,"-")
    op = idx_minus >= 0
    idx_op = (1-op)*idx_plus+op*idx_minus
    op1 = [eval(all_op[i][:idx_op[i]]) for i in range(len(all_op))]
    op2 = [eval(all_op[i][idx_op[i]+1:]) for i in range(len(all_op))]
    idx = np.lexsort(np.stack([op2,op1,res,op]))

    # -------------------------------
    # Display
    # -------------------------------
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    ylim = (-0.1, 1.1)
    print_title('same two operands NRS')
    ax_A = letter('A',plot_by_layer)(f, gs[0,0], scales, same_operand_rs[[0,2,6,8],:,best_step_idx], ylim = ylim, ylabel = f'two same\noperands NRS', title = ['V1', 'V2', 'V3', 'IPS'], correlate = True, xticks = scales[::4], xlabel = 'gain $G$')
    ax_B = letter('B',multiplot_by_layer)(f, gs[0,1], np.arange(4), same_operand_rs[[0,2,6,8],:,best_step_idx].T, scales, xlabel = 'layer', ylabel = f'two same\noperands NRS', ylim = ylim , title = '', xlim = (-0.5,3.5), xticklabels = ['V1', 'V2', 'V3', 'IPS'], showbar = True, clabel = 'gain $G$', cticks = np.arange(1,6))
    print_title('same left operand NRS')
    ax_C = letter('C',plot_by_layer)(f, gs[1,0], scales, left_same_operand_rs[[0,2,6,8],:,best_step_idx], ylim = ylim, ylabel = f'same left\noperand NRS', title = ['V1', 'V2', 'V3', 'IPS'], correlate = True, xticks = scales[::4], xlabel = 'gain $G$')
    ax_D = letter('D',multiplot_by_layer)(f, gs[1,1], np.arange(4), left_same_operand_rs[[0,2,6,8],:,best_step_idx].T, scales, xlabel = 'layer', ylabel = f'same left\noperand NRS', ylim = ylim , title = '', xlim = (-0.5,3.5), xticklabels = ['V1', 'V2', 'V3', 'IPS'], showbar = True, clabel = 'gain $G$', cticks = np.arange(1,6))
    print_title('same right operand NRS')
    ax_E = letter('E',plot_by_layer)(f, gs[2,0], scales, right_same_operand_rs[[0,2,6,8],:,best_step_idx], ylim = ylim, ylabel = f'same right\noperand NRS', title = ['V1', 'V2', 'V3', 'IPS'], correlate = True, xticks = scales[::4], xlabel = 'gain $G$')
    ax_F = letter('F',multiplot_by_layer)(f, gs[2,1], np.arange(4), right_same_operand_rs[[0,2,6,8],:,best_step_idx].T, scales, xlabel = 'layer', ylabel = f'same right\noperand NRS', ylim = ylim , title = '', xlim = (-0.5,3.5), xticklabels = ['V1', 'V2', 'V3', 'IPS'], showbar = True, clabel = 'gain $G$', cticks = np.arange(1,6))
    print_title('two different operands NRS')
    ax_G = letter('G',plot_by_layer)(f, gs[3,0], scales, different_operand_rs[[0,2,6,8],:,best_step_idx], ylim = ylim, ylabel = f'two different\noperands NRS', title = ['V1', 'V2', 'V3', 'IPS'], correlate = True, xticks = scales[::4], xlabel = 'gain $G$')
    ax_H = letter('H',multiplot_by_layer)(f, gs[3,1], np.arange(4), different_operand_rs[[0,2,6,8],:,best_step_idx].T, scales, xlabel = 'layer', ylabel = f'two different\noperands NRS', ylim = ylim , title = '', xlim = (-0.5,3.5), xticklabels = ['V1', 'V2', 'V3', 'IPS'], showbar = True, clabel = 'gain $G$', cticks = np.arange(1,6))
    f.savefig(f'{figure_path}/png/figureS2.png', dpi = 1200)
    f.savefig(f'{figure_path}/pdf/figureS2.pdf', dpi = 1200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure S7 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)
