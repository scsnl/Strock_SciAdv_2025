import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from tqdm import tqdm

def correlation(rsa_c, mean_a, k, times, ctimes):
    for l in range(times[k]):
        rsa_c[ctimes[k]+l] = np.corrcoef(mean_a[:,l])

def distance(rsa_d, mean_a, k, times, ctimes, ctimes2):
    rsa_d[ctimes[k]:ctimes2[k]] = np.linalg.norm(mean_a[:, None]-mean_a[None,:], axis = -1).T

def main(args):

    seed = 0
    np.random.seed(seed)

    # -------------------------------
    # Parameters
    # -------------------------------

    n_max = 18
    scales = np.linspace(1.0, 5.0, 17)
    steps = np.arange(0, 3801, 100)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path containing activities
    activity_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/activity'
    # path containing tuning of neurons
    tuning_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/tuning'
    # path where rsa is saved
    rsa_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/rsa_test'
    os.makedirs(rsa_path, exist_ok=True)
    ext = f'_{args.variant}' if len(args.variant) > 0 else ''


    # -------------------------------
    # Display
    # -------------------------------
    task = np.load(f'{activity_path}/task.npz')
    label = task["label"]
    op = task["op"][:,0]
    all_op = task["all_op"]
    add_op = np.where(np.char.find(all_op,"+") >= 0)[0]
    sub_op = np.where(np.char.find(all_op,"-") >= 0)[0]
    add_idx = np.isin(op, add_op)
    sub_idx = np.isin(op, sub_op)
    modules = ['V1', 'V2', 'V4', 'IT']
    times = np.array([1,2,4,2])
    ctimes2 = np.cumsum(times)
    ctimes = ctimes2 - times
    if args.similarity == 'correlation':
        update_rsa = lambda rsa,mean_a,k: correlation(rsa, mean_a, k, times, ctimes)
    elif args.similarity == 'distance':
        update_rsa = lambda rsa,mean_a,k: distance(rsa, mean_a, k, times, ctimes, ctimes2)
    filename = [f'{rsa_path}/rsa_{args.similarity}_scaled{ext}.npy']
    if not np.all([os.path.exists(f) for f in filename]) or args.redo:
        rsa = np.nan*np.empty((len(steps), len(scales), np.sum(times), len(all_op), len(all_op)))
        for i, step in enumerate(tqdm(steps)):
            for j, scale in enumerate(tqdm(scales, leave = False)):
                for k, m in enumerate(pbar := tqdm(modules, leave = False)):
                    try:
                        if scale%0.5 == 0:
                            model_name = f'scaled_{scale:.1f}{ext}'
                        else:
                            model_name =  f'scaled_{scale:.2f}{ext}'
                        d = f'{tuning_path}/{model_name}/step{step:05}'
                        f = f'{d}/{m}.npy'
                        if not os.path.exists(f) or args.redo_tuning:
                            pbar.set_description(f'Computing tuning curve')
                            a = np.load(f'{activity_path}/{model_name}/step{step:05}/{m}.npz')[m]
                            os.makedirs(d, exist_ok=True)
                            mean_a = np.empty((len(all_op),)+a.shape[1:])
                            for o in range(len(all_op)):
                                mean_a[o] = np.mean(a[op == o], axis = 0)
                            del a
                            np.save(f, mean_a)
                        else:
                            pbar.set_description(f'Loading tuning curve')
                            mean_a = np.load(f)
                        pbar.set_description(f'Reshaping')
                        mean_a = mean_a.reshape((mean_a.shape[0:2])+(-1,))
                        pbar.set_description(f'Computing {args.similarity}')
                        update_rsa(rsa[i,j], mean_a, k)
                    except:
                        print(f'Redo G={scale:.2f} step={step:d}')
        np.save(filename[0], rsa)
    else:
        rsa = np.load(filename[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Compute similarity between addition and subtraction representation')
    parser.add_argument('--variant', metavar = 'V', type = str, default = 'fixedbatchnorm', help = 'variant used')
    parser.add_argument('--similarity', metavar = 'V', type = str, default = 'correlation', help = 'variant used')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--redo_tuning', action='store_true')
    args = parser.parse_args()
    main(args)