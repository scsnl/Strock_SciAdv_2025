import torch
import pytorch_lightning as pl
from nn_modeling.model.torch import ScaledCORnet
from nn_modeling.dataset.arithmetic import ArithmeticDataset
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import pandas as pd
import os, psutil
process = psutil.Process(os.getpid())
from humanfriendly import format_size
import argparse
from functools import reduce
import sys
from torch import nn

def main(args):

    seed = 0
    pl.seed_everything(seed)

    # -------------------------------
    # Parameters
    # -------------------------------

    n_max = 18
    n_sample_train = 50 # number of samples used in training per class/condition
    n_sample_test = 50 # number of samples used in test per class/condition
    sample = np.arange(n_sample_train, n_sample_train+n_sample_test) # id of sample used
    if args.modules is None:
        modules = ['V1','V2','V4','IT']
        ext = ''
    elif len(args.modules) > 0:
        modules = args.modules
        ext = '_'.join(['']+modules)
    else:
        raise NameError('At least one module (i.e. V1, V2, V4 or IT) has to be hyper-excitable')

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    model_name = f'scaled{ext}_{args.scale[0]}{f"_rec_1_{args.rec[0]}_{args.rec[1]}_{args.rec[2]}" if not args.rec is None else ""}_fixedbatchnorm'
    if args.pretrained:
        model_path = f'{model_path}_pretrained'
    print(f'Model: {model_name}')
    # path where activities are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/activity/{model_name}'
    # path where inputs and desired outputs are saved
    task_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/activity'
    # path where accuracies are saved
    accuracy_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/accuracy/{model_name}'
    # path where log of test are saved
    log_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/log/test'
    # path containing the dataset
    dataset_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/stimuli'
    # path containing the model
    model_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/model/{model_name}'

    # -------------------------------
    # Test dataset
    # -------------------------------

    # only the selected sample are used
    pattern_sample = f'\\d*_\\d*(\\+|-)\\d*_({reduce(lambda a,b: f"{a}|{b}", sample)})'
    # simple global pattern containing files
    path = f'{dataset_path}/*.png'
    # regex matching only the files that should be used for training
    include = f'{dataset_path}/{pattern_sample}.png'

    dataset = ArithmeticDataset(path = path, summary = f'{dataset_path}/test.pkl', include = include)
    test_loader = DataLoader(dataset, batch_size = 100, shuffle = False, num_workers = 10, pin_memory = True)

    # -------------------------------
    # Saving model
    # -------------------------------

    def hook_output(m, i, o):
        activity[m].append(o.cpu())

    class LabelConditionCallback(Callback):
        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            label.append(batch[1].cpu())
            op.append(batch[2].cpu())

    accuracy = np.zeros(len(args.step))
    trainer = pl.Trainer(default_root_dir=log_path, callbacks = [LabelConditionCallback()], deterministic=True, devices="auto", accelerator="auto")
    model  = ScaledCORnet(out_features = n_max+1, scale = args.scale[0], times = args.rec, modules = modules, pretrained = args.pretrained, fixed_batchnorm = True)

    if args.saveall:
        modules_l = ["V1", "V2", "V4", "IT", "decoder"]
    else:
        modules_l = ["decoder"]

    modules = [getattr(model.model, m).output for m in modules_l]
    module_names = {getattr(model.model, m).output:m for m in modules_l}
    times = {getattr(model.model, m).output:getattr(model.model, m).times if hasattr(getattr(model.model, m), "times") else 1 for m in modules_l}
    for m in modules:
        m.register_forward_hook(hook_output)

    # -------------------------------
    # Testing model
    # -------------------------------

    for i, step in enumerate(args.step):
        os.makedirs(f'{activity_path}/step{step:05}', exist_ok = True)
        checkpoint = torch.load(f'{model_path}/step{step:05}.ckpt')
        model.load_state_dict(checkpoint['state_dict'])
        activity = {}
        label = []
        op = []
        for m in modules:
            activity[m]= []
        metrics, = trainer.test(model, test_loader)
        label = torch.cat(label).cpu().numpy()[:, None]
        op = torch.cat(op).cpu().numpy()[:, None]
        if not os.path.exists(f'{task_path}/task.npz') or args.redo:
            np.savez_compressed(f'{task_path}/task.npz', label = label, op = op, all_op = dataset.all_op)
            print(f'label, op and all_op saved in .npz ({format_size(process.memory_info().rss)})')
        else:
            print(f'label, op and all_op already saved in .npz ({format_size(process.memory_info().rss)})')
        accuracy[i] = metrics['test_acc_epoch']
        for m in modules:
            if not os.path.exists(f'{activity_path}/step{step:05}/{module_names[m]}.npz') or args.redo:
                print(f'starting saving {module_names[m]} at step {step} ({format_size(process.memory_info().rss)})')
                tmp = torch.stack([torch.cat(activity[m][i::times[m]]) for i in range(times[m])], axis = 1).numpy()
                print(f'tmp created ({format_size(process.memory_info().rss)})')
                del activity[m]
                print(f'activity[m] removed ({format_size(process.memory_info().rss)})')
                print(module_names[m], tmp.shape)
                data_dict = {}
                data_dict[module_names[m]] = tmp
                print(f'data_dict created ({format_size(process.memory_info().rss)})')
                np.savez_compressed(f'{activity_path}/step{step:05}/{module_names[m]}.npz', **data_dict)
                print(f'{module_names[m]} at step {step} saved in .npz ({format_size(process.memory_info().rss)})')
                del tmp
                del data_dict
            else:
                print(f'{module_names[m]} at step {step} already saved in .npz ({format_size(process.memory_info().rss)})')

    step_accuracy = np.zeros(len(args.step), dtype = np.dtype([('step', np.int64, 1), ('accuracy', np.float64, 1)]))
    step_accuracy['step'] = args.step
    step_accuracy['accuracy'] = accuracy
    os.makedirs(f'{accuracy_path}', exist_ok=True)
    np.save(f'{accuracy_path}/steps_{np.min(args.step)}_{np.max(args.step)}_accuracy.npy', step_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test models at different steps')
    parser.add_argument('--step', metavar = 'E', type = int, nargs = "+", help = 'list of steps to test')
    parser.add_argument('--scale', metavar='S', type = float, nargs=1, help='scale used')
    parser.add_argument('--rec', metavar='S', type = int, nargs=3, default = None, help='rec time used for V2 V4 IT')
    parser.add_argument('--saveall', action='store_true', help='If set, saves internal activity, else save only decoder')
    parser.add_argument('--modules', metavar='M', type = str, nargs='*', default = None, help='which modules are hyper-excitable (default V1, V2, V4 and IT)')
    parser.add_argument('--pretrained', action = 'store_true', help='if set use CORnet pretrained on ImageNet')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)