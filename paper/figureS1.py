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
    best_step = 700
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
    ws = phi*np.array([1,1,1,1])
    wspace = phi/2
    hs = np.array([1,1,1])
    hspace = phi

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 0.5, right = 1)

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

    os.makedirs(f'{figure_path}/figure1', exist_ok=True)
    for i in [1, 7]:
        for j, name in zip([0,2,6,8], ['V1', 'V2', 'V3', 'IPS']):
            f = plt.figure(figsize = (2,2))
            ax = f.add_subplot(1,1,1)
            ax.imshow(rs[6,i,j,idx][:,idx], zorder = 0, vmin = -1, vmax = 1, cmap = 'viridis')#, cmap = 'coolwarm')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(Rectangle((np.sum(op)-0.5, -0.5), np.sum(1-op)-0.5, np.sum(op)-0.5, linewidth=3, edgecolor='C3', facecolor='none', zorder = 10, clip_on = False))
            f.savefig(f'{figure_path}/figure1/D_g_{scales[i]:.1f}_{j:d}_{name}.png', dpi = 600)
    
    params = np.stack([op,res,op1,op2])
    # -------------------------------
    # Display
    # -------------------------------
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    print_title('add-sub NRS')

    xlim = (-100,3900)
    ylim = (-0.1, 1.1)
    ax_A = letter('A',plot_value_by_iteration)(f, gs[0,0], addsub_rs[0][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V1')
    letter('',plot_value_by_iteration)(f, gs[0,1], addsub_rs[2][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V2')
    letter('',plot_value_by_iteration)(f, gs[0,2], addsub_rs[6][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V3')
    letter('',plot_value_by_iteration)(f, gs[0,3], addsub_rs[8][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), title = 'IPS')
    ax_B = letter('B',plot_value_by_iteration)(f, gs[1,0], addadd_rs[0][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-add NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V1')
    letter('',plot_value_by_iteration)(f, gs[1,1], addadd_rs[2][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-add NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V2')
    letter('',plot_value_by_iteration)(f, gs[1,2], addadd_rs[6][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-add NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V3')
    letter('',plot_value_by_iteration)(f, gs[1,3], addadd_rs[8][:, selected_steps_idx], scales, selected_steps, ylabel = 'add-add NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), title = 'IPS')
    ax_C = letter('C',plot_value_by_iteration)(f, gs[2,0], subsub_rs[0][:, selected_steps_idx], scales, selected_steps, ylabel = 'sub-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V1')
    letter('',plot_value_by_iteration)(f, gs[2,1], subsub_rs[2][:, selected_steps_idx], scales, selected_steps, ylabel = 'sub-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V2')
    letter('',plot_value_by_iteration)(f, gs[2,2], subsub_rs[6][:, selected_steps_idx], scales, selected_steps, ylabel = 'sub-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), showbar = False, title = 'V3')
    letter('',plot_value_by_iteration)(f, gs[2,3], subsub_rs[8][:, selected_steps_idx], scales, selected_steps, ylabel = 'sub-sub NRS', clabel = 'gain $G$', ylim = ylim, xlim = xlim, cticks = np.arange(1,6), title = 'IPS')
    f.savefig(f'{figure_path}/png/figureS1.png', dpi = 1200)
    f.savefig(f'{figure_path}/pdf/figureS1.pdf', dpi = 1200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure S1 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)
