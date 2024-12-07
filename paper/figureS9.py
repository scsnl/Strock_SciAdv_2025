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
from utils.plots import get_figsize, letter, plot_bar, violinplot, plot_value_by_iteration, plot_bar_by_gain, plot, barplot_by_layer, scatter

def main(args):


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
    selected_steps = np.arange(100, 3801, 100)
    selected_steps_idx = np.where(selected_steps[None,:] == steps[:,None])[0]
    lag = 30
    ext = f'_fixedbatchnorm'

    idx_roi = 1
    name_roi = 'right IPL/IPS'
    thresholds = [0.95]
    modules = ['V1', 'V2', 'V3', 'IPS']
    idx_modules = modules_idx = [0,2,6,8]

    # -------------------------------
    # Figure parameters
    # -------------------------------

    sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.get_cmap("coolwarm"))
    c_random = '0.7'
    c_td = sm.to_rgba(0)
    c_md = sm.to_rgba(1)

    zoom = 10
    groups = ['TD', 'MD']
    phi = (1+np.sqrt(5))/2
    ws = phi*np.array([1,1])
    wspace = phi
    hs = np.array([1,1,1,1])
    hspace = 1

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # Path where accuracy are saved
    accuracy_path = f'{os.environ.get("DATA_PATH")}/{task}/accuracy'
    # Path where model activity are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/{task}/activity'
    # Path where representational similarity are saved
    rs_path = f'{os.environ.get("DATA_PATH")}/{task}/rsa_test'
    # Path where patient behavior are saved
    behavior_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/behavior'
    # Path where distance between model and patient are saved
    distance_path = f'{os.environ.get("DATA_PATH")}/{task}/distance/accuracy'
    # Path were entropy are saved
    distribution_path = f'{os.environ.get("DATA_PATH")}/{task}/distribution'
    entropy_path = f'{os.environ.get("DATA_PATH")}/{task}/entropy'
    # Path containing tuning
    manifold_path = f'{os.environ.get("DATA_PATH")}/{task}/manifold' if args.dataset in ['h', 'f'] else f'{os.environ.get("DATA_PATH")}/{task}/manifold_proj'
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

    # Model add-sub similarity

    task = np.load(f'{activity_path}/task.npz')
    rs = np.load(f'{rs_path}/rsa_correlation_scaled{ext}.npy')
    all_op = task["all_op"]
    all_add = np.where(np.char.find(all_op,"+") >= 0)[0]
    all_sub = np.where(np.char.find(all_op,"-") >= 0)[0]
    addsub_rs = np.mean(rs[:,:,:,all_add][:,:,:,:,all_sub], axis = (3,4)).T
    addadd_rs = np.mean(rs[:,:,:,all_add][:,:,:,:,all_add], axis = (3,4)).T
    subsub_rs = np.mean(rs[:,:,:,all_sub][:,:,:,:,all_sub], axis = (3,4)).T

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

    # Model number line precision

    h_b = np.exp(np.load(f'{entropy_path}/response.npy'))
    all_n = np.load(f'{distribution_path}/n_summary_scaled{ext}.npy')
    all_op = np.load(f'{activity_path}/task.npz')["all_op"]
    label = np.array([eval(op) for op in all_op])
    label_n = np.zeros(all_n.shape[:-2]+(n_max+1,n_max+1))
    for i in range(n_max+1):
        idx = label == i
        label_n[:,:,i] = np.sum(all_n[:,:,idx], axis = 2)
    p = label_n/np.sum(label_n, axis = -1, keepdims = True)
    mean_response = np.sum(np.arange(n_max+1)*p, axis = -1)
    trueness_response = np.mean(np.abs(mean_response-np.arange(n_max+1)[None,None]), axis = -1)
    distance = np.abs(np.arange(n_max+1)[:,None]-np.arange(n_max+1)[None,:])
    precision_response = np.mean(np.sqrt(np.sum(((np.arange(n_max+1)-mean_response[:,:,:,None])**2)*p, axis = -1)), axis = -1)

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

    # Matching
    
    path = [f'{distance_path}/best_{var}{ext}_numopsv2.npy' for var in ['excitability']+[f"distance{bn}" for bn in ["b"]]]
    best_e_numops = np.load(path[0])
    best_db_numops = np.load(path[1])
    best_step_b_idx = np.argmin(np.sum(best_db_numops, axis = 0))
    best_step_b = steps[best_step_b_idx]
    best_e = best_e_numops[:,best_step_b_idx]
    best_e_idx = np.where(scales[None] == best_e[:,None])[1]
    best_step_b_selected_idx = np.where(selected_steps == best_step_b)[0][0]

    # Model manifold properties

    capacity = []
    radius = []
    dimension = []
    correlation = []

    for i,step in enumerate(tqdm(selected_steps[1:])):
        try:
            capacity.append(np.load(f'{manifold_path}/manifold_capacity{ext}_step{step:02}.npz')['arr_0'])
            radius.append(np.load(f'{manifold_path}/manifold_radius{ext}_step{step:02}.npz')['arr_0'])
            dimension.append(np.load(f'{manifold_path}/manifold_dimension{ext}_step{step:02}.npz')['arr_0'])
            correlation.append(np.load(f'{manifold_path}/manifold_correlation{ext}_step{step:02}.npz')['arr_0'])
        except: # TODO: Remove after all done
            print(f'Redo manifold analysis step {step:d}')
            capacity.append(np.nan*capacity[-1])
            radius.append(np.nan*radius[-1])
            dimension.append(np.nan*dimension[-1])
            correlation.append(np.nan*correlation[-1])
    capacity = np.stack(capacity)
    radius = np.stack(radius)
    dimension = np.stack(dimension)
    correlation = np.stack(correlation)

    capacity = (1/np.mean(1/capacity, axis = -1)).T
    radius = np.mean(radius, axis = -1).T
    dimension = np.mean(dimension, axis = -1).T
    correlation = correlation.T

    # Normalize data

    print(best_step_b_idx, best_e_idx)
    min_acc, max_acc = patient_data['addsub_accuracy'].min(), patient_data['addsub_accuracy'].max()
    transform_acc = lambda x: min_acc+x*(max_acc-min_acc)
    acc[...] = (acc-min_acc)/(max_acc-min_acc)
    min_addsub_rs, max_addsub_rs = addsub_rs.min(axis = (1,2), keepdims = True), addsub_rs.max(axis = (1,2), keepdims = True)
    transform_addsub_rs = lambda x: min_addsub_rs+x*(max_addsub_rs-min_addsub_rs)
    idx = ['addsub_accuracy', 'numops', name_roi]
    patient_data[idx] = (patient_data[idx]-patient_data[idx].min())/(patient_data[idx].max()-patient_data[idx].min())

    # Matched behavioral/neural

    addsub_rs[...] = (addsub_rs-min_addsub_rs)/(max_addsub_rs-min_addsub_rs)
    addsub_rs_best_e = addsub_rs[:, best_e_idx, best_step_b_idx][idx_modules]
    addsub_rs_best_e_td = np.percentile(addsub_rs_best_e[:,patient_data['Group']=='TD'], q = [5, 50, 95], axis = 1)
    addsub_rs_best_e_md = np.percentile(addsub_rs_best_e[:,patient_data['Group']=='MD'], q = [5, 50, 95], axis = 1)
    addsub_rs_child = patient_data[name_roi]
    trueness_response_best_e = trueness_response[best_e_idx]
    precision_response_best_e = precision_response[best_e_idx]
    h_b_best_e = h_b[best_e_idx]
    correlation_best_e = correlation[:, best_e_idx, best_step_b_selected_idx]
    correlation_best_e_td = np.percentile(correlation_best_e[:,patient_data['Group']=='TD'], q = [5, 50, 95], axis = 1)
    correlation_best_e_md = np.percentile(correlation_best_e[:,patient_data['Group']=='MD'], q = [5, 50, 95], axis = 1)

    # Prediction of patient with best model addsub-similarity best matched with numops

    lagged_predicted_patient_b_numops = acc[best_e_idx, best_step_b_idx:best_step_b_idx+lag]
    lagged_predicted_patient_b2_numops = trueness_response[best_e_idx, best_step_b_idx:best_step_b_idx+lag]
    lagged_predicted_patient_b3_numops = precision_response[best_e_idx, best_step_b_idx:best_step_b_idx+lag]
    lagged_predicted_patient_b4_numops = h_b[best_e_idx, best_step_b_idx:best_step_b_idx+lag]
    lagged_predicted_patient_n_numops = addsub_rs[:,best_e_idx,best_step_b_idx:best_step_b_idx+lag]
    lagged_predicted_patient_n0_numops = addadd_rs[:,best_e_idx,best_step_b_idx:best_step_b_idx+lag]
    lagged_predicted_patient_n1_numops = subsub_rs[:,best_e_idx,best_step_b_idx:best_step_b_idx+lag]
    lagged_predicted_patient_n2_numops = capacity[:,best_e_idx,best_step_b_idx-1:best_step_b_idx+lag-1]
    lagged_predicted_patient_n3_numops = dimension[:,best_e_idx,best_step_b_idx-1:best_step_b_idx+lag-1]
    lagged_predicted_patient_n4_numops = correlation[:,best_e_idx,best_step_b_idx-1:best_step_b_idx+lag-1]
    idx = {g:(patients["Group"]==g).to_numpy() for g in groups}

    size_effect = np.median(lagged_predicted_patient_b_numops[idx['MD']], axis = 0)-np.median(lagged_predicted_patient_b_numops[idx['TD'],:1], axis = 0)
    #size_effect[size_effect<0] = np.inf
    #size_effect[size_effect>0] = -np.inf
    best_lag = np.argmin(np.abs(size_effect))
    md_step_idx = best_step_b_idx + best_lag
    md_step = steps[md_step_idx]

    # Time for MD to catch-up to TD

    catchup = np.empty((len(scales), len(selected_steps)))
    median_acc_td = np.median(acc[best_e_idx, best_step_b_idx:best_step_b_idx+1][idx['TD']])
    for i in range(len(scales)):
        for j in range(len(selected_steps)):
            size_effect = np.abs(median_acc_td-acc[i, selected_steps_idx][j:]) if acc[i,j]<median_acc_td else [0]
            catchup[i,j] = 100*np.argmin(size_effect)

    additional_training = 100*catchup[best_e_idx, best_step_b_idx]/(best_step_b+1)
    total_training = 100*(catchup[best_e_idx, best_step_b_idx]+best_step_b)/(best_step_b+1)

    # -------------------------------
    # Debug
    # -------------------------------

    print_title('Data shapes')
    print(f'acc: {acc.shape}')
    print(f'rs: {addsub_rs.shape}, {addsub_rs[-1,:,:].shape}')
    print_title('Figure')
    print(f'figsize: {figsize}')

    # -------------------------------
    # Stats
    # -------------------------------

    print_title('Stats')
    print(f'Best-matching iteration {1+best_step_b}')
    print(f'MD catch-up with TD at iteration {1+md_step}')
    print(f'Average additional training for TD: {np.mean((100*catchup[:,best_step_b_idx]/(best_step_b+1))[best_e_idx[idx["TD"]]]):.2f}%')
    print(f'Average additional training for MD: {np.mean((100*catchup[:,best_step_b_idx]/(best_step_b+1))[best_e_idx[idx["MD"]]]):.2f}%')

    # -------------------------------
    # Display
    # -------------------------------

    first_step_idx = selected_steps_idx[0]
    last_step_idx = selected_steps_idx[-1]

    print_title('Display')
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    ax_2A = letter('2A',plot_value_by_iteration)(f, gs[0,0], transform_acc(acc[:,selected_steps_idx]), scales, selected_steps, ylabel = 'accuracy', clabel = 'gain $G$', ylim = (-0.1, 1.1), cticks = np.arange(1,6), threshold = 0.95, chance = 1/19)
    ax_3C = letter('3C',violinplot)(f, gs[0,1], [best_e[patient_data['Group']=='MD'], best_e[patient_data['Group']=='TD']] , names = ['MLD\npDNN','TD\npDNN'], xlabel = '', ylabel = f'gain $G$', ylim = (0.5, 5.5), c = [c_md, c_td], pd = 'all')
    ax_5A = letter('5A', barplot_by_layer)(f, gs[1,0], modules, [addsub_rs_best_e_md, addsub_rs_best_e_td], c = [c_md, c_td], label = ['MLD', 'TD'], ylabel = 'add-sub NRS')
    ax_5B = letter('5B', scatter)(f, gs[1,1], addsub_rs_child, addsub_rs_best_e[-1], xlabel = 'children add-sub NRS', ylabel = 'pDNN add-sub NRS', correlate = True)
    ax_6E = letter('6E',violinplot)(f, gs[2,0], [trueness_response_best_e[patient_data['Group']=='MD', best_step_b_idx], trueness_response_best_e[patient_data['Group']=='TD', best_step_b_idx]] , names = ['MLD\npDNN','TD\npDNN'], ylim = (-0.1, 1.1), c = [c_md, c_td], pd = 'all', ylabel = 'numerical systematic error')
    ax_6G = letter('6G',violinplot)(f, gs[2,1], [h_b_best_e[patient_data['Group']=='MD', best_step_b_idx], h_b_best_e[patient_data['Group']=='TD', best_step_b_idx]] , names = ['MLD\npDNN','TD\npDNN'], ylim = (-1, 20), c = [c_md, c_td], pd = 'all', ylabel = '# different responses')
    ax_7F = letter('7F', barplot_by_layer)(f, gs[3,0], modules, [correlation_best_e_md[:,modules_idx], correlation_best_e_td[:,modules_idx]], c = [c_md, c_td], label = ['MLD', 'TD'], ylabel = 'inter-manifold correlation')
    ax_8B = letter('8B',violinplot)(f, gs[3,1], [additional_training[patient_data['Group']=='MD'], additional_training[patient_data['Group']=='TD']] , names = ['MLD','TD'], ylim = (-10, 150), c = [c_md, c_td], pd = [(0,1)], ylabel = 'additional training (%)')#, title = None)
    f.savefig(f'{figure_path}/png/figure_S9.png', dpi = 1200)
    f.savefig(f'{figure_path}/pdf/figure_S9.pdf', dpi = 1200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure S9 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h+f', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)
