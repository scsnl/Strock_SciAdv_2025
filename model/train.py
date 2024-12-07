import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nn_modeling.dataset.arithmetic import ArithmeticDataset
from nn_modeling.model.torch import ScaledCORnet
import numpy as np
from functools import reduce
import argparse
import sys
from torch import nn

def main(args):
	seed = 0
	pl.seed_everything(seed)

	# -------------------------------
	# Parameters
	# -------------------------------

	n_max = 18 # maximal result of operation
	n_sample = 50 # number of samples used in training per class
	sample = np.arange(n_sample) # id of sample used
	n_epoch = 20
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

	if args.dataset == 'h':
		tasks = [f'addsub_{n_max}']
	elif args.dataset == 'f':
		tasks =  [f'addsub_{n_max}_font']
	elif args.dataset == 'h+f' or args.dataset == 'f+h':
		tasks = [f'addsub_{n_max}{s}' for s in ['', '_font']]
	task =  '+'.join(tasks)

	# path where model are saved
	model_path = f'{os.environ.get("DATA_PATH")}/{task}/model/scaled{ext}_{args.scale[0]}{f"_rec_1_{args.rec[0]}_{args.rec[1]}_{args.rec[2]}" if not args.rec is None else ""}'
	model_path = f'{model_path}_fixedbatchnorm'
	if args.pretrained:
		model_path = f'{model_path}_pretrained'

	os.makedirs(f'{model_path}', exist_ok=True)
	# path where log of training are saved
	log_path = f'{os.environ.get("DATA_PATH")}/{task}/log/train'
	# path containing the dataset
	dataset_path = [f'{os.environ.get("DATA_PATH")}/{task}/stimuli' for task in tasks]

	# -------------------------------
	# Training dataset
	# -------------------------------

	# only the selected sample are used
	pattern_sample = f'\\d*_\\d*(\\+|-)\\d*_({reduce(lambda a,b: f"{a}|{b}", sample)})'
	# simple global pattern containing files
	path = [f'{p}/*.png' for p in dataset_path]
	# regex matching only the files that should be used for training
	include = [f'{p}/{pattern_sample}.png' for p in dataset_path]

	os.makedirs(f'{os.environ.get("DATA_PATH")}/{task}/stimuli', exist_ok=True)
	dataset = ArithmeticDataset(path = path, summary = f'{os.environ.get("DATA_PATH")}/{task}/stimuli/train.pkl',include = include)
	train_loader = DataLoader(dataset, batch_size = 100, shuffle = True, num_workers = 10, pin_memory = True)

	# -------------------------------
	# Initializing model
	# -------------------------------

	model  = ScaledCORnet(out_features = n_max+1, scale = args.scale[0], times = args.rec, modules = modules, pretrained = args.pretrained, fixed_batchnorm = True)

	# -------------------------------
	# Saving model
	# -------------------------------

	# saving initial model
	torch.save({
		"epoch": 0,
		"global_step": 0,
		"pytorch-lightning_version": pl.__version__,
		"state_dict": model.state_dict()
	}, f'{model_path}/step{0:05d}.ckpt')
	# using checkpoint to save models after each epoch
	checkpoint = pl.callbacks.ModelCheckpoint(dirpath = model_path, filename = 'step{step:05d}', auto_insert_metric_name = False, save_on_train_epoch_end = True, save_top_k = -1, every_n_train_steps = 100)
	# saving gpu stats
	gpu_stats = pl.callbacks.DeviceStatsMonitor()

	# -------------------------------
	# Training model
	# -------------------------------

	trainer = pl.Trainer(default_root_dir = log_path, callbacks = [gpu_stats, checkpoint], deterministic = True, gpus = 1, max_epochs=n_epoch)
	trainer.fit(model, train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train to solve arithmetic task with scaled CORnet')
    parser.add_argument('--scale', metavar='S', type = float, nargs=1, default = [1.0], help='scale used')
    parser.add_argument('--rec', metavar='S', type = int, nargs=3, default = None, help='rec time used for V2 V4 IT')
    parser.add_argument('--modules', metavar='M', type = str, nargs='*', default = None, help='which modules are hyper-excitable (default V1, V2, V4 and IT)')
    parser.add_argument('--pretrained', action = 'store_true', help='if set use CORnet pretrained on ImageNet')
    parser.add_argument('--dataset', metavar='D', type = str, default = 'h', choices = ['h', 'f', 'h+f'], help='Which dataset is used to train')
    args = parser.parse_args()
    main(args)