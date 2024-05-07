import torch
import pytorch_lightning as pl
from common.model.mycornet import ScaledCORnet
from torchsummary import summary
import argparse
import os
import sys

def main(args):

    seed = 0
    pl.seed_everything(seed)

    # -------------------------------
    # Parameters
    # -------------------------------

    n_max = 18

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path containing the model
    model_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/model/scaled_{args.scale[0]}'

    # -------------------------------
    # Model
    # -------------------------------

    model  = ScaledCORnet(out_features = n_max+1, scale = args.shift[0], times = args.rec,  pretrained = False)

    # -------------------------------
    # Testing model
    # -------------------------------

    for i, step in enumerate(args.step):
        checkpoint = torch.load(f'{model_path}/step{step:05}.ckpt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print(summary(model, (3,28,140)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Print summary of model')
    parser.add_argument('--step', metavar = 'E', type = int, nargs = "+", help = 'list of steps to test')
    parser.add_argument('--scale', metavar='S', type = float, nargs=1, default = [1.0], help='scale used')
    parser.add_argument('--rec', metavar='S', type = int, nargs=3, default = None, help='rec time used for V2 V4 IT')
    args = parser.parse_args()
    main(args)