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
from matplotlib.colors import Normalize
from utils.plots import get_figsize, letter, plot_bar, violinplot

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
    # Path where patient behavior are saved
    behavior_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/behavior'
    # Path where distance between model and patient are saved
    distance_path = f'{os.environ.get("DATA_PATH")}/{task}/distance/accuracy'
    os.makedirs(distance_path, exist_ok=True)
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

    # Patient behavioral data

    patients = pd.read_csv(f'{behavior_path}/TD_MD.csv')
    patient_data = pd.read_csv(f'{os.environ.get("OAK")}/projects/ehk/dnn-modeling-v2/raw/TD_MD_NPs.csv')
    patient_data = patient_data.set_index('PID')
    patient_data = patient_data.reindex(index=patients['PID'].values)
    patient_data = patient_data.reset_index()
    patient_data['numops'] = patient_data['number_operations_standard:wiat ii']
    patient_data['addsub_accuracy'] = np.nan
    
    for i, p in enumerate(pbar := tqdm(range(patients.shape[0]), leave = False)):
        patient_id = patients["PID"][p]
        patient_group = patients["Group"][p]
        pbar.set_description(f'Processing kid {patient_group} {patient_id:3d}')
        behavior = pd.read_csv(f'{behavior_path}/{patient_id}.csv')
        patient_data.at[i, 'addsub_accuracy'] = behavior_acc(behavior)

    # Normalize data

    min_acc, max_acc = acc.min(), acc.max()
    min_acc, max_acc = patient_data['addsub_accuracy'].min(), patient_data['addsub_accuracy'].max()
    transform_acc = lambda x: min_acc+x*(max_acc-min_acc)
    acc[...] = (acc-min_acc)/(max_acc-min_acc)
    idx = ['addsub_accuracy', 'numops']
    patient_data[idx] = (patient_data[idx]-patient_data[idx].min())/(patient_data[idx].max()-patient_data[idx].min())
    
    # Distance model accuracy vs patient numops

    path = [f'{distance_path}/best_{var}{ext}_numopsv2.npy' for var in ['excitability']+[f"distance{bn}" for bn in ["b"]]]
    if not np.all([os.path.exists(f) for f in path]) or True:
        best_e_numops = np.empty((len(patients), len(selected_steps)))
        best_db_numops = np.empty((len(patients), len(selected_steps)))
        for i, p in enumerate(pbar := tqdm(range(patients.shape[0]), leave = False)):
            db = np.abs(acc[:,selected_steps_idx] - patient_data['numops'][i])
            idx = np.argmin(db, axis = 0)
            best_db_numops[i] = db[idx, np.arange(len(selected_steps))]
            best_e_numops[i] = scales[idx]
        np.save(path[0], best_e_numops)
        np.save(path[1], best_db_numops)
    else:
        best_e_numops = np.load(path[0])
        best_db_numops = np.load(path[1])

    # Prediction of patient with model matched with numops
    
    predicted_patient_b_numops = np.empty((len(patient_data), len(selected_steps)))
    idx_best_e, idx_patient, idx_time = np.where(best_e_numops[None, :, :39] == scales[:, None, None])
    predicted_patient_b_numops[idx_patient, idx_time] = acc[idx_best_e, idx_time]

    rs_numops_b = np.empty((len(selected_steps)))
    ps_numops_b = np.empty((len(selected_steps)))
    for i, j in enumerate(selected_steps_idx):
        rs_numops_b[i], ps_numops_b[i] = pearsonr(predicted_patient_b_numops[:,i], patient_data['numops'].values)
    rs_numops_b[np.isnan(rs_numops_b)] = 0
    ps_numops_b[np.isnan(ps_numops_b)] = 1

    # Random matching

    n_per = 100
    path = [f'{distance_path}/best_{var}{ext}_random.npy' for var in ['excitability']+[f"distance{bn}" for bn in ["b"]]]
    if not np.all([os.path.exists(f) for f in path]) or True:
        predicted_patient_b_random = np.empty((n_per, len(patient_data), len(selected_steps)))
        best_e_random = np.empty((n_per, len(patients), len(selected_steps)))
        best_db_random = np.empty((n_per, len(patients), len(selected_steps)))
        for i, p in enumerate(pbar := tqdm(range(patients.shape[0]), leave = False)):
            db = np.abs(acc[:,selected_steps_idx] - patient_data['numops'][i])
            for j in range(n_per):
                idx = np.random.randint(len(scales), size = (len(selected_steps),))
                predicted_patient_b_random[j,i] = acc[idx, np.arange(len(selected_steps))]
                best_db_random[j,i] = db[idx, np.arange(len(selected_steps))]
                best_e_random[j,i] = scales[idx]
        np.save(path[0], best_e_random)
        np.save(path[1], best_db_random)
    else:
        best_e_random = np.load(path[0])
        best_db_random = np.load(path[1])

    # Random prediction

    rs_random_b = np.empty((n_per,len(selected_steps)))
    ps_random_b = np.empty((n_per,len(selected_steps)))
    for k in range(n_per):
        for i, j in enumerate(selected_steps_idx):
            rs_random_b[k,i], ps_random_b[k,i] = pearsonr(predicted_patient_b_random[k,:,i], patient_data['numops'].values)
    rs_random_b[np.isnan(rs_random_b)] = 0
    ps_random_b[np.isnan(ps_random_b)] = 1


    # Matching

    best_step_b_idx = np.argmin(np.sum(best_db_numops, axis = 0))
    best_step_b = steps[best_step_b_idx]
    best_e = best_e_numops[:,best_step_b_idx]
    best_e_idx = np.where(scales[None] == best_e[:,None])[1]
    acc_best_e = transform_acc(acc[best_e_idx, best_step_b_idx])
    acc_child = transform_acc(patient_data['numops'])

    # -------------------------------
    # Stats
    # -------------------------------
    
    print_title('Debug')
    for k in ['age', 'fsiq:wasi', 'reading_comp_standard:wiat ii', 'number_operations_standard:wiat ii']:
        for g in groups:
            _X = patient_data[k][patient_data['Group']==g]
            print(f'{k} ({g}): {np.mean(_X):.2f}±{np.std(_X, ddof = 1):.2f}')
        print(f'{k} (all): {np.mean(patient_data[k]):.2f}±{np.std(patient_data[k], ddof = 1):.2f}')
        print(ttest_ind(patient_data[k][patient_data['Group']=='MD'], patient_data[k][patient_data['Group']=='TD']))
    print(f'acc: {acc.shape}')
    print(f'figsize: {figsize}')
    print(f'best matching iteration: {1+best_step_b}')
    print_title('Stats')

    print(f"Size effect distance MLD vs distance TD: {cohen_d(best_db_numops[patient_data['Group']=='MD',best_step_b_idx], best_db_numops[patient_data['Group']=='TD',best_step_b_idx])}")
    
    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    ax_A = letter('A',plot_bar)(f, gs[0,0], [np.percentile(best_db_random, q = [5,50,95], axis = (0,1)), np.percentile(best_db_numops[patient_data['Group']=='TD'], q = [5,50,95], axis = 0), np.percentile(best_db_numops[patient_data['Group']=='MD'], q = [5,50,95], axis = 0)], selected_steps, c = [c_random, c_td, c_md], ylabel = 'behavioral distance', best_step = best_step_b,  ylim = (-0.1, 1.1), label = ['Random', 'TD', 'MLD'])
    ax_B = letter('B',violinplot)(f, gs[0,1], [best_db_numops[patient_data['Group']=='MD',best_step_b_idx], best_db_numops[patient_data['Group']=='TD',best_step_b_idx], best_db_random[:,:,best_step_b_idx].flatten()] , names = ['MLD','TD','Random'], xlabel = '', ylabel = f'behavioral distance', ylim = (-0.1, 1.1), c = [c_md, c_td, c_random], pd = 'all')
    ax_C = letter('C',violinplot)(f, gs[1,0], [best_e[patient_data['Group']=='MD'], best_e[patient_data['Group']=='TD']] , names = ['MLD\npDNN','TD\npDNN'], xlabel = '', ylabel = f'gain $G$', ylim = (0.5, 5.5), c = [c_md, c_td], pd = 'all')
    ax_D = letter('D',violinplot)(f, gs[1,1], [acc_best_e[patient_data['Group']=='MD'], acc_child[patient_data['Group']=='MD'], acc_best_e[patient_data['Group']=='TD'], acc_child[patient_data['Group']=='TD']] , names = ['MLD\npDNN', 'MLD\nChildren', 'TD\npDNN', 'TD\nChildren'], xlabel = '', ylabel = f'behavioral score', ylim = (-0.1, 1.1), c = [c_md, c_md, c_td, c_td], pd = [(0,1), (2,3), [(0,2), (0,3), (1,2), (1,3)]], printstats = [(0,2), (1,3)])
    f.savefig(f'{figure_path}/png/figure3.png', dpi = 1200)
    f.savefig(f'{figure_path}/pdf/figure3.pdf', dpi = 1200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 3 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)
