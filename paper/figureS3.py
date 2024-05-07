# General import
import os, sys, argparse
from tqdm import tqdm

# Data import
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, ttest_1samp, rv_discrete
from utils.data import behavior_acc, matching_ranges
from utils.prompt import print_title

# Plot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from utils.plots import get_figsize, letter, plot_value_by_iteration, plot_value_by_gain, plot, plot_bar, plot_r_by_iteration, violinplot, imshow_by_gain

def main(args):

    seed = 0
    np.random.seed(seed)

    # -------------------------------
    # Data parameters
    # -------------------------------

    n_max = 18
    scales = np.linspace(1.0, 5.0, 17)
    steps = np.arange(0, 3801, 100)
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
    ws = phi*np.array([1,1,1])
    wspace = phi
    hs = np.array([1])
    hspace = phi

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 1.5, right = 2)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # Path where accuracy are saved
    accuracy_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/accuracy'
    # Path where model activity are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/activity'
    # Path where patient behavior are saved
    behavior_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/behavior'
    # Path where distance between model and patient are saved
    distance_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/distance/accuracy'
    # Path were entropy are saved
    entropy_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/entropy'
    distribution_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/distribution'
    # Path where figure are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper'
    os.makedirs(figure_path, exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    # Model accuracy

    acc = np.empty((len(scales), len(steps)))
    for i, scale in enumerate(tqdm(scales, leave = False)):
        if scale%0.5 == 0:
            accuracy = np.load(f'{accuracy_path}/scaled_{scale:.1f}{ext}/steps_0_3800_accuracy.npy')
        else:
            accuracy = np.load(f'{accuracy_path}/scaled_{scale:.2f}{ext}/steps_0_3800_accuracy.npy')
        idx = np.where(accuracy["step"][:,None] == steps[None, :])[0]
        acc[i] = accuracy["accuracy"][idx]

    # Model accuracy per item

    all_n = np.load(f'{distribution_path}/n_summary_scaled{ext}.npy')
    all_op = np.load(f'{activity_path}/task.npz')["all_op"]
    label = np.array([eval(op) for op in all_op])
    label_n = np.zeros(all_n.shape[:-2]+(n_max+1,n_max+1)) # gain x step x expected x provided
    for i in range(n_max+1):
        idx = label == i
        label_n[:,:,i] = np.sum(all_n[:,:,idx], axis = 2)

    # Matching
    
    path = [f'{distance_path}/best_{var}{ext}_numopsv2.npy' for var in ['excitability']+[f"distance{bn}" for bn in ["b"]]]
    best_e_numops = np.load(path[0])
    best_db_numops = np.load(path[1])
    best_step_b_idx = np.argmin(np.sum(best_db_numops, axis = 0))
    best_step_b = steps[best_step_b_idx]
    best_e = best_e_numops[:,best_step_b_idx]
    best_e_idx = np.where(scales[None] == best_e[:,None])[1]

    # Model numerical precision
    p = label_n/np.sum(label_n, axis = -1, keepdims = True) # gain x step x expected x provided
    mean_response = np.sum(np.arange(n_max+1)*p, axis = -1)
    trueness_response = np.mean(np.abs(mean_response-np.arange(n_max+1)[None,None]), axis = -1)
    precision_response = np.mean(np.sqrt(np.sum(((np.arange(n_max+1)-mean_response[:,:,:,None])**2)*p, axis = -1)), axis = -1)
    trueness_response_best_e = trueness_response[best_e_idx]
    precision_response_best_e = precision_response[best_e_idx]

    h_b = np.exp(np.load(f'{entropy_path}/response.npy'))
    h_b_best_e = h_b[best_e_idx]


    # Model 

    patients = pd.read_csv(f'{behavior_path}/TD_MD.csv')
    patient_data = pd.read_csv(f'{os.environ.get("OAK")}/projects/ehk/dnn-modeling-v2/raw/TD_MD_NPs.csv')
    patient_data = patient_data.set_index('PID')
    patient_data = patient_data.reindex(index=patients['PID'].values)


    # -------------------------------
    # Stats
    # -------------------------------
    
    print_title('Debug')
    print(f'acc shape: {acc.shape}')
    print(f'figsize: {figsize}')
    print_title('Stats')
    print(f'Best accuracy iteration {1+steps[0]:d}: {np.max(acc[:,0]):.3f}')
    print(f'Worse accuracy iteration {1+selected_steps[-1]:d}: {np.min(acc[:,selected_steps_idx[-1]]):.3f}')
    print(f'Worse accuracy iteration {1+steps[-1]:d}: {np.min(acc[:,-1]):.3f}')
    print(f'Correlation accuracy vs numerical trueness: {pearsonr(acc[:, selected_steps_idx].flatten(), trueness_response.flatten())}')
    print(f'Correlation accuracy vs numerical precision: {pearsonr(acc[:, selected_steps_idx].flatten(), precision_response.flatten())}')
    print(f'Correlation accuracy vs entropy: {pearsonr(acc[:, selected_steps_idx].flatten(), h_b.flatten())}')
    print(f'Estimated number of values used at best matching: {h_b[::2, best_step_b_idx]}')
    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))

    ax_A = letter('A',plot_value_by_iteration)(f, gs[0,0], trueness_response[:,selected_steps_idx], scales, selected_steps, ylabel = 'numerical trueness', clabel = 'gain $G$', ylim = (-0.1,5.1), cticks = np.arange(1,6))
    ax_B = letter('B',plot_value_by_iteration)(f, gs[0,1], precision_response[:,selected_steps_idx], scales, selected_steps, ylabel = 'numerical precision', clabel = 'gain $G$', ylim = (-0.1, 5.1), cticks = np.arange(1,6))
    ax_C = letter('C',plot_value_by_iteration)(f, gs[0,2], h_b[:,selected_steps_idx], scales, selected_steps, ylabel = '# different responses', clabel = 'gain $G$', ylim = (-2, 20), cticks = np.arange(1,6))
    
    f.savefig(f'{figure_path}/figureS3.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 6 of manuscript')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
