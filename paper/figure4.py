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
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, rgb2hex
from utils.plots import get_figsize, letter, plot_value_by_iteration, plot_bar_by_gain, plot_by_layer, multiplot_by_layer, violinplot_by_gain, plot, imshow_by_layer

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
    best_step = 1100
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
    hs = np.array([phi,phi,1,1,1])
    hspace = phi

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 1.5, right = 2)

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
    all_add = np.where(np.char.find(all_op,"+") >= 0)[0]
    all_sub = np.where(np.char.find(all_op,"-") >= 0)[0]
    addsub_rs = np.mean(rs[:,:,:,all_add][:,:,:,:,all_sub], axis = (3,4)).T
    addadd_rs = np.mean(rs[:,:,:,all_add][:,:,:,:,all_add], axis = (3,4)).T
    subsub_rs = np.mean(rs[:,:,:,all_sub][:,:,:,:,all_sub], axis = (3,4)).T
    overall_rs = np.mean(rs, axis = (3,4)).T
    within_rs = np.mean([addadd_rs, subsub_rs], axis = 0)
    increase_rs = within_rs-addsub_rs

    rs_rs = np.empty((addsub_rs.shape[0],len(selected_steps)))
    ps_rs = np.empty((addsub_rs.shape[0],len(selected_steps)))
    std_rs = np.empty((addsub_rs.shape[0],len(selected_steps)))
    for i, j in enumerate(selected_steps_idx):
        for k in range(addsub_rs.shape[0]):
            try:
                rs_rs[k,i], ps_rs[k,i] = pearsonr(addsub_rs[k,:,j], scales)
            except:
                rs_rs[k,i], ps_rs[k,i] = np.nan, np.nan
            std_rs[k,i] = np.std(addsub_rs[k,:,j])

    # -------------------------------
    # Stats
    # -------------------------------
    
    print_title('Debug')
    print(f'rs shape: {rs.shape}')
    print(f'addsub rs shape: {addsub_rs.shape}')
    print(f'figsize: {figsize}')

    print_title('Stats')
    r,p = pearsonr(np.tile(scales[:,None], (1,)+(addsub_rs.shape[2:])).flatten(), addsub_rs[0].flatten())
    print(f'Correlation gain $G$ vs add-sub NRS (V1, ALL): (r = {r:.3f}, p = {p:.2e})')
    r,p = pearsonr(np.tile(scales[:,None], (1,)+(addsub_rs.shape[2:])).flatten(), addsub_rs[2].flatten())
    print(f'Correlation gain $G$ vs add-sub NRS (V2, ALL): (r = {r:.3f}, p = {p:.2e})')
    r,p = pearsonr(np.tile(scales[:,None], (1,)+(addsub_rs.shape[2:])).flatten(), addsub_rs[6].flatten())
    print(f'Correlation gain $G$ vs add-sub NRS (V3, ALL): (r = {r:.3f}, p = {p:.2e})')
    r,p = pearsonr(np.tile(scales[:,None], (1,)+(addsub_rs.shape[2:])).flatten(), addsub_rs[8].flatten())
    print(f'Correlation gain $G$ vs add-sub NRS (IPS, ALL): (r = {r:.3f}, p = {p:.2e})')

    cmap = 'viridis'
    norm  = Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    print(f'Color1: + {rgb2hex(sm.to_rgba(addadd_rs[8,0,5]))}, - {rgb2hex(sm.to_rgba(subsub_rs[8,0,5]))}, +/- {rgb2hex(sm.to_rgba(addsub_rs[8,0,5]))}')
    print(f'Color2: + {rgb2hex(sm.to_rgba(addadd_rs[8,0,6]))}, - {rgb2hex(sm.to_rgba(subsub_rs[8,0,6]))}, +/- {rgb2hex(sm.to_rgba(addsub_rs[8,0,6]))}')
    
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
    ax_A = letter('A',imshow_by_layer)(f, gs[0,0], rs[best_step_idx,5,[0,2,6,8]][:,idx][:,:,idx], title = ['V1', 'V2', 'V3', 'IPS'], xlabel = 'operation', ylabel = 'operation', clabel = 'similarity', vlim = (-1, 1))
    ax_B = letter('B',imshow_by_layer)(f, gs[1,0], rs[best_step_idx,12,[0,2,6,8]][:,idx][:,:,idx], title = ['V1', 'V2', 'V3', 'IPS'], xlabel = 'operation', ylabel = 'operation', clabel = 'similarity', vlim = (-1, 1))
    print_title('add-sub NRS')
    ax_C = letter('C',plot_by_layer)(f, gs[2,0], scales, addsub_rs[[0,2,6,8],:,best_step_idx], ylim = ylim, ylabel = f'add-sub NRS', title = ['V1', 'V2', 'V3', 'IPS'], correlate = True, xticks = scales[::4], xlabel = 'gain $G$')
    ax_D = letter('D',multiplot_by_layer)(f, gs[2,1], np.arange(4), addsub_rs[[0,2,6,8],:,best_step_idx].T, scales, xlabel = 'layer', ylabel = f'add-sub NRS', ylim = ylim , title = '', xlim = (-0.5,3.5), xticklabels = ['V1', 'V2', 'V3', 'IPS'], showbar = True, clabel = 'gain $G$', cticks = np.arange(1,6))
    print_title('add-add NRS')
    ax_E = letter('E',plot_by_layer)(f, gs[3,0], scales, addadd_rs[[0,2,6,8],:,best_step_idx], ylim = ylim, ylabel = f'add-add NRS', title = ['V1', 'V2', 'V3', 'IPS'], correlate = True, xticks = scales[::4], xlabel = 'gain $G$')
    ax_F = letter('F',multiplot_by_layer)(f, gs[3,1], np.arange(4), addadd_rs[[0,2,6,8],:,best_step_idx].T, scales, xlabel = 'layer', ylabel = f'add-add NRS', ylim = ylim , title = '', xlim = (-0.5,3.5), xticklabels = ['V1', 'V2', 'V3', 'IPS'], showbar = True, clabel = 'gain $G$', cticks = np.arange(1,6))
    print_title('sub-sub NRS')
    ax_G = letter('G',plot_by_layer)(f, gs[4,0], scales, subsub_rs[[0,2,6,8],:,best_step_idx], ylim = ylim, ylabel = f'sub-sub NRS', title = ['V1', 'V2', 'V3', 'IPS'], correlate = True, xticks = scales[::4], xlabel = 'gain $G$')
    ax_H = letter('H',multiplot_by_layer)(f, gs[4,1], np.arange(4), subsub_rs[[0,2,6,8],:,best_step_idx].T, scales, xlabel = 'layer', ylabel = f'sub-sub NRS', ylim = ylim , title = '', xlim = (-0.5,3.5), xticklabels = ['V1', 'V2', 'V3', 'IPS'], showbar = True, clabel = 'gain $G$', cticks = np.arange(1,6))
    f.savefig(f'{figure_path}/png/figure4.png', dpi = 1200)
    f.savefig(f'{figure_path}/pdf/figure4.pdf', dpi = 1200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 4 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)
