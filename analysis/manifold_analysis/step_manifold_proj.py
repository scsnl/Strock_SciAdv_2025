import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from tqdm import tqdm
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from humanfriendly import format_size
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from packages.neural_manifolds_replicaMFT.mftma.manifold_analysis_correlation import manifold_analysis_corr
from sklearn.model_selection import train_test_split

def plot_components(c, ev, p, p2, s, xlabel, ylabel, clabel, xlim = None, ylim = None):
    n_c = c.shape[-1]
    n_s = c.shape[0]
    f = plt.figure(figsize = (2*n_c, 2*n_s), constrained_layout = True)
    cmap = cm.get_cmap("viridis")
    norm  = Normalize(vmin=np.min(p2), vmax=np.max(p2))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i in range(n_c):
        for j in range(n_s):
            ax = f.add_subplot(n_s, n_c, j*n_c+i+1)
            ax.scatter(c[j,:,i], p, s = 10, c = p2, alpha = 0.1, linewidth = 0)
            ax.set_xlabel(f'{xlabel}{i} ({ev[j,i]:.1%})')
            if not xlim is None:
                ax.set_xlim(*xlim)
            if not ylim is None:
                ax.set_xlim(*ylim)
            if i == 0:
                ax.set_ylabel(f'$G = {s[j]}$\n{ylabel}')
    f.colorbar(sm, ax = ax, label = clabel)
    return f

def main(args):

    seed = 0
    np.random.seed(seed)

    # -------------------------------
    # Parameters
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
    steps = args.steps
    n_c = 100

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path containing activities
    activity_path = f'{os.environ.get("DATA_PATH")}/{task}/activity'
    # path where manifold variables are saved
    pca_path = f'{os.environ.get("DATA_PATH")}/{task}/manifold_proj'
    os.makedirs(pca_path, exist_ok=True)
    
    ext = f'_{args.variant}' if len(args.variant) > 0 else ''

    # -------------------------------
    # Compute manifold geometry
    # -------------------------------

    task = np.load(f'{activity_path}/task.npz')
    op = task["op"][:,0]
    all_op = task["all_op"]
    label = np.array([eval(op) for op in all_op])[op]

    modules = ['V1', 'V2', 'V4', 'IT']
    times = np.array([1,2,4,2])
    ctimes2 = np.cumsum(times)
    ctimes = ctimes2 - times
    _, idx = train_test_split(np.arange(len(op)), test_size=0.5, stratify=op)

    for i, step in enumerate(tqdm(steps)):
        filename = [f'{pca_path}/manifold{s}{ext}_step{step:02}.npz' for s in ['_capacity', '_radius', '_dimension', '_correlation']]
        if not np.all([os.path.exists(f) for f in filename]) or args.redo:
            capacity = np.nan*np.empty((len(scales), np.sum(times), n_max+1))
            radius = np.nan*np.empty((len(scales), np.sum(times), n_max+1))
            dimension = np.nan*np.empty((len(scales), np.sum(times), n_max+1))
            correlation = np.nan*np.empty((len(scales), np.sum(times)))
            for j, scale in enumerate(tqdm(scales, leave = False)):
                if scale%0.5 == 0:
                    model_name = f'scaled_{scale:.1f}{ext}'
                else:
                    model_name =  f'scaled_{scale:.2f}{ext}'
                for k, m in enumerate(tqdm(modules, leave = False)):
                    _a = np.load(f'{activity_path}/{model_name}/step{step:05}/{m}.npz')[m]
                    for l in range(times[k]):
                        a = _a[:,l].reshape((_a.shape[0],-1))
                        X = [a[idx][label[idx]==res].T for res in range(n_max+1)]
                        n, p = X[0].shape
                        #if n > 1000:
                        #    P = np.random.normal(loc = 0, scale = 1, size = (1000, n))
                        #    P /= np.sqrt(np.sum(P*P, axis=1, keepdims=True))
                        #    X = [P@x for x in X]
                        capacity[j,ctimes[k]+l], radius[j,ctimes[k]+l], dimension[j,ctimes[k]+l], correlation[j,ctimes[k]+l], _ = manifold_analysis_corr(X, 0, 100)
            np.savez(filename[0], capacity)
            np.savez(filename[1], radius)
            np.savez(filename[2], dimension)
            np.savez(filename[3], correlation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Compute manifold geometrical properties (using projection for less computation time)')
    parser.add_argument('--variant', metavar = 'V', type = str, default = 'fixedbatchnorm', help = 'variant used')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--steps', metavar = 'E', type = int, nargs = "+", help = 'list of steps to test')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)