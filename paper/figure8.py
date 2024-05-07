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
from utils.plots import get_figsize, letter, plot_bar, violinplot, plot_value_by_iteration, plot_bar_by_gain, plot

def main(args):


    # -------------------------------
    # Data parameters
    # -------------------------------

    n_max = 18
    scales = np.linspace(1.0, 5.0, 17)
    steps = np.arange(0, 3801, 100)
    selected_steps = np.arange(100, 3801, 100)
    selected_steps_idx = np.where(selected_steps[None,:] == steps[:,None])[0]
    lag = 30
    ext = f'_fixedbatchnorm'

    idx_roi = 1
    name_roi = 'right IPL/IPS'

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
    ws = phi*np.array([1,1,1])
    wspace = phi
    hs = np.array([1,1,1,1])
    hspace = phi

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 1, right = 1.5)

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
    # Path were entropy are saved
    distribution_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/distribution'
    entropy_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/entropy'
    # Path containing tuning
    manifold_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/manifold'
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

    # Model manifold properties

    capacity = []
    radius = []
    dimension = []
    correlation = []

    for i,step in enumerate(tqdm(selected_steps[1:])):
        capacity.append(np.load(f'{manifold_path}/manifold_capacity{ext}_step{step:02}.npz')['arr_0'])
        radius.append(np.load(f'{manifold_path}/manifold_radius{ext}_step{step:02}.npz')['arr_0'])
        dimension.append(np.load(f'{manifold_path}/manifold_dimension{ext}_step{step:02}.npz')['arr_0'])
        correlation.append(np.load(f'{manifold_path}/manifold_correlation{ext}_step{step:02}.npz')['arr_0'])
    capacity = np.stack(capacity)
    radius = np.stack(radius)
    dimension = np.stack(dimension)
    correlation = np.stack(correlation)

    capacity = (1/np.mean(1/capacity, axis = -1)).T
    radius = np.mean(radius, axis = -1).T
    dimension = np.mean(dimension, axis = -1).T
    correlation = correlation.T

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

    best_lag = np.where(ttest_ind(lagged_predicted_patient_b_numops[idx['MD']], lagged_predicted_patient_b_numops[idx['TD'],:1], axis = 0, alternative = 'less')[1] > 0.05)[0][0]
    size_effect = np.median(lagged_predicted_patient_b_numops[idx['MD']], axis = 0)-np.median(lagged_predicted_patient_b_numops[idx['TD'],:1], axis = 0)
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
    print('HERE:', additional_training)
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
    ax_A = letter('A',plot)(f, gs[0,0], scales, catchup[:,best_step_b_idx], xlabel = 'gain $G$', ylabel = f'catching iteration', xlim = (0.5, 5.5), xticks = scales[::4], ylim = (-200, 1700), loc = 'lower right', correlate = True)
    ax_B = letter('B',violinplot)(f, gs[0,1], [additional_training[patient_data['Group']=='MD'], additional_training[patient_data['Group']=='TD']] , names = ['MLD','TD'], ylim = (-10, 150), c = [c_md, c_td], pd = [(0,1)], ylabel = 'additional training (%)')#, title = None)
    ax_B = letter('C',violinplot)(f, gs[0,2], [lagged_predicted_patient_b_numops[patient_data['Group']=='MD', 0], lagged_predicted_patient_b_numops[patient_data['Group']=='MD', best_lag], lagged_predicted_patient_b_numops[patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 1.0), c = [c_md, c_md, c_td, c_td], pd = [(0,1),(1,2)], ylabel = 'accuracy')
    ax_C = letter('D',violinplot)(f, gs[1,0], [lagged_predicted_patient_b2_numops[patient_data['Group']=='MD', 0], lagged_predicted_patient_b2_numops[patient_data['Group']=='MD', best_lag], lagged_predicted_patient_b2_numops[patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 2.1), c = [c_md, c_md, c_td, c_td], pd = [(1,2),(0,1)], ylabel = 'numerical systematic error')
    ax_D = letter('E',violinplot)(f, gs[1,1], [lagged_predicted_patient_b3_numops[patient_data['Group']=='MD', 0], lagged_predicted_patient_b3_numops[patient_data['Group']=='MD', best_lag], lagged_predicted_patient_b3_numops[patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 2.1), c = [c_md, c_md, c_td, c_td], pd = [(1,2),(0,1)], ylabel = 'numerical imprecision')
    ax_D = letter('F',violinplot)(f, gs[1,2], [lagged_predicted_patient_b4_numops[patient_data['Group']=='MD', 0], lagged_predicted_patient_b4_numops[patient_data['Group']=='MD', best_lag], lagged_predicted_patient_b4_numops[patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 20.1), c = [c_md, c_md, c_td, c_td], pd = [(0,1),(1,2)], ylabel = '# different responses')
    ax_E = letter('G',violinplot)(f, gs[2,0], [lagged_predicted_patient_n_numops[8,patient_data['Group']=='MD', 0], lagged_predicted_patient_n_numops[8,patient_data['Group']=='MD', best_lag], lagged_predicted_patient_n_numops[8,patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 1.0), c = [c_md, c_md, c_td, c_td], pd = [(1,2),(0,1)], ylabel = 'add-sub NRS')
    ax_E = letter('H',violinplot)(f, gs[2,1], [lagged_predicted_patient_n0_numops[8,patient_data['Group']=='MD', 0], lagged_predicted_patient_n0_numops[8,patient_data['Group']=='MD', best_lag], lagged_predicted_patient_n0_numops[8,patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 1.0), c = [c_md, c_md, c_td, c_td], pd = [(1,2),(0,1)], ylabel = 'add-add NRS')
    ax_E = letter('I',violinplot)(f, gs[2,2], [lagged_predicted_patient_n1_numops[8,patient_data['Group']=='MD', 0], lagged_predicted_patient_n1_numops[8,patient_data['Group']=='MD', best_lag], lagged_predicted_patient_n1_numops[8,patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 1.0), c = [c_md, c_md, c_td, c_td], pd = [(1,2),(0,1)], ylabel = 'sub-sub NRS')
    ax_F = letter('J',violinplot)(f, gs[3,0], [lagged_predicted_patient_n2_numops[-1,patient_data['Group']=='MD', 0], lagged_predicted_patient_n2_numops[-1,patient_data['Group']=='MD', best_lag], lagged_predicted_patient_n2_numops[-1,patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.01, 0.07), c = [c_md, c_md, c_td, c_td], pd = [(0,1),(1,2)], ylabel = 'manifold capacity')
    ax_G = letter('K',violinplot)(f, gs[3,1], [lagged_predicted_patient_n3_numops[-1,patient_data['Group']=='MD', 0], lagged_predicted_patient_n3_numops[-1,patient_data['Group']=='MD', best_lag], lagged_predicted_patient_n3_numops[-1,patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-1, 201), c = [c_md, c_md, c_td, c_td], pd = [(1,2),(0,1)], ylabel = 'manifold dimensionality')
    ax_H = letter('L',violinplot)(f, gs[3,2], [lagged_predicted_patient_n4_numops[-1,patient_data['Group']=='MD', 0], lagged_predicted_patient_n4_numops[-1,patient_data['Group']=='MD', best_lag], lagged_predicted_patient_n4_numops[-1,patient_data['Group']=='TD', 0]] , names = ['MLD\nPre','MLD\nPost','TD\nPre'], ylim = (-0.1, 0.9), c = [c_md, c_md, c_td, c_td], pd = [(1,2),(0,1)], ylabel = 'inter-manifold correlation')
    f.savefig(f'{figure_path}/figure8.png', dpi = 200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 8 of manuscript')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
