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
from utils.plots import get_figsize, letter, plot_value_by_iteration, plot_value_by_gain, plot, plot_bar, plot_r_by_iteration, violinplot, violinplot_by_gain, plot_by_layer, barplot_by_layer, scatter, hist

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
    modules = ['V1', 'V2', 'V3', 'IPS']
    idx_modules = [0,2,6,8] 

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
    hspace = 1.5

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 1.5)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # Path where accuracy are saved
    accuracy_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/accuracy'
    # Path where model activity are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/activity'
    # Path where representational similarity are saved
    rs_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/rsa_test'
    # Path where patient behavior are saved
    behavior_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/behavior'
    # Path where distance between model and patient are saved
    distance_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/distance/accuracy'
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
    increase_rs = within_rs - addsub_rs

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

    # Patient behavioral data

    patients = pd.read_csv(f'{behavior_path}/TD_MD.csv')
    patient_data = pd.read_csv(f'{os.environ.get("OAK")}/projects/ehk/dnn-modeling-v2/raw/TD_MD_NPs.csv')
    patient_data = patient_data.set_index('PID')
    patient_data = patient_data.reindex(index=patients['PID'].values)
    patient_data = patient_data.reset_index()
    patient_data.rename(columns = {'number_operations_standard:wiat ii':'numops'}, inplace = True)
    patient_data['addsub_accuracy'] = np.nan
    
    for i, p in enumerate(pbar := tqdm(range(patients.shape[0]), leave = False)):
        patient_id = patients["PID"][p]
        patient_group = patients["Group"][p]
        pbar.set_description(f'Processing kid {patient_group} {patient_id:3d}')
        behavior = pd.read_csv(f'{behavior_path}/{patient_id}.csv')
        patient_data.at[i, 'addsub_accuracy'] = behavior_acc(behavior)

    # Patient add-sub similarity

    patient_addsub_rs = pd.read_csv(f'{os.environ.get("OAK")}/projects/lchen32/2019_TD_MD_AddSub_Similarity/scripts/taskfmri/RSA/roi_rsa_signal_addcom-addsim_VS_subcom-subsim_21MD24TD_oneGrpPlustwoGrpInter_withV1.tsv', sep='\t')
    patient_addsub_rs = patient_addsub_rs.set_index('Subject')
    patient_addsub_rs = patient_addsub_rs.reindex(index=patients['PID'].values)
    patient_addsub_rs = patient_addsub_rs.reset_index()
    patient_data[name_roi] = np.tanh(patient_addsub_rs.values[:,3+idx_roi]) # z -> r

    # Normalize data

    min_acc, max_acc = patient_data['addsub_accuracy'].min(), patient_data['addsub_accuracy'].max()
    transform_acc = lambda x: min_acc+x*(max_acc-min_acc)
    acc[...] = (acc-min_acc)/(max_acc-min_acc)
    min_addsub_rs, max_addsub_rs = addsub_rs.min(axis = (1,2), keepdims = True), addsub_rs.max(axis = (1,2), keepdims = True)
    transform_addsub_rs = lambda x: min_addsub_rs+x*(max_addsub_rs-min_addsub_rs)
    addsub_rs[...] = (addsub_rs-min_addsub_rs)/(max_addsub_rs-min_addsub_rs)

    idx = ['addsub_accuracy', 'numops', name_roi]
    patient_data[idx] = (patient_data[idx]-patient_data[idx].min())/(patient_data[idx].max()-patient_data[idx].min())

    # Distance model accuracy vs patient numops

    path = [f'{distance_path}/best_{var}{ext}_numopsv2.npy' for var in ['excitability']+[f"distance{bn}" for bn in ["b"]]]
    best_e_numops = np.load(path[0])
    best_db_numops = np.load(path[1])

    # Matching

    best_step_b_idx = np.argmin(np.sum(best_db_numops, axis = 0))
    best_step_b = steps[best_step_b_idx]
    best_e = best_e_numops[:,best_step_b_idx]
    best_e_idx = np.where(scales[None] == best_e[:,None])[1]
    acc_best_e = transform_acc(acc[best_e_idx, best_step_b_idx])
    acc_child = transform_acc(patient_data['numops'])

    addsub_rs_best_e = addsub_rs[:, best_e_idx, best_step_b_idx][idx_modules]
    addsub_rs_best_e_td = np.percentile(addsub_rs_best_e[:,patient_data['Group']=='TD'], q = [5, 50, 95], axis = 1)
    addsub_rs_best_e_md = np.percentile(addsub_rs_best_e[:,patient_data['Group']=='MD'], q = [5, 50, 95], axis = 1)
    addsub_rs_child = patient_data[name_roi]

    addadd_rs_best_e = addadd_rs[:, best_e_idx, best_step_b_idx][idx_modules]
    addadd_rs_best_e_td = np.percentile(addadd_rs_best_e[:,patient_data['Group']=='TD'], q = [5, 50, 95], axis = 1)
    addadd_rs_best_e_md = np.percentile(addadd_rs_best_e[:,patient_data['Group']=='MD'], q = [5, 50, 95], axis = 1)

    subsub_rs_best_e = subsub_rs[:, best_e_idx, best_step_b_idx][idx_modules]
    subsub_rs_best_e_td = np.percentile(subsub_rs_best_e[:,patient_data['Group']=='TD'], q = [5, 50, 95], axis = 1)
    subsub_rs_best_e_md = np.percentile(subsub_rs_best_e[:,patient_data['Group']=='MD'], q = [5, 50, 95], axis = 1)

    increase_rs_best_e = increase_rs[:, best_e_idx, best_step_b_idx][idx_modules]
    increase_rs_best_e_td = np.percentile(increase_rs_best_e[:,patient_data['Group']=='TD'], q = [5, 50, 95], axis = 1)
    increase_rs_best_e_md = np.percentile(increase_rs_best_e[:,patient_data['Group']=='MD'], q = [5, 50, 95], axis = 1)

    # Random matching

    r0,p0 = pearsonr(addsub_rs_best_e[-1], addsub_rs_child)
    path = [f'{distance_path}/best_{var}{ext}_random.npy' for var in ['excitability']+[f"distance{bn}" for bn in ["b"]]]
    best_e_random = np.load(path[0])[:,:,best_step_b_idx]
    rs,ps = np.zeros(best_e_random.shape[0]), np.zeros(best_e_random.shape[0])
    for i in range(best_e_random.shape[0]):
        best_e_random_idx = np.where(scales[None] == best_e_random[i,:,None])[1]
        rs[i], ps[i] = pearsonr(addsub_rs[-1, best_e_random_idx, best_step_b_idx], addsub_rs_child)

    # -------------------------------
    # Stats
    # -------------------------------st
    
    
    print_title('Debug')
    print(f'acc shape: {acc.shape}')
    print(f'addsub rs shape: {addsub_rs.shape}')
    print(f'figsize: {figsize}')
    print(f'best matching iteration: {1+best_step_b}')
    print_title('Stats')
    t,p = ttest_1samp(rs, r0)
    print(f'Prediction add-sub NRS random vs b matching: (t = {t:.3f}, p = {p:.2e})')
    print(f"Size effect add-sub NRS: {cohen_d(addsub_rs_best_e[:,patient_data['Group']=='TD'], addsub_rs_best_e[:,patient_data['Group']=='MD'], axis = 1)}")
    print_title('add-sub NRS')
    for i, m in  enumerate(modules):
        t,p = ttest_ind(addsub_rs_best_e[i][patient_data['Group']=='MD'], addsub_rs_best_e[i][patient_data['Group']=='TD'])
        print(f'add-sub NRS MD pDNN vs TD pDNN ({m}): (t = {t:.3f}, p = {p:.2e})')

    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    ax_A = letter('A', barplot_by_layer)(f, gs[0,0], modules, [addsub_rs_best_e_md, addsub_rs_best_e_td], c = [c_md, c_td], label = ['MLD', 'TD'], ylabel = 'add-sub NRS')
    ax_B = letter('B', scatter)(f, gs[0,1], addsub_rs_child, addsub_rs_best_e[-1], xlabel = 'children add-sub NRS', ylabel = 'pDNN add-sub NRS', correlate = True)
    ax_C = letter('C', hist)(f, gs[0,2], rs, r0, xlabel = 'r')
    f.savefig(f'{figure_path}/figure5.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 5 of manuscript')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
