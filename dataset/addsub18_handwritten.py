import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nn_modeling.dataset.arithmetic import *
from inspect import signature
from itertools import product
from torchvision.datasets import MNIST
import freetype as ft
import sys

seed = 0
np.random.seed(seed)

# -------------------------------
# General parameters
# -------------------------------

n_max = 18 # maximal result of operation
ns = np.arange(n_max+1) # number considered in the operations
char_dim = (28,28) # dimension of one character
n_sample = 100 # number of sample per numerosity

# -------------------------------
# Operation parameters
# -------------------------------

fmt = int_fmt(2)
var = Var(fmt)
op = [var+var, var-var]
op = [(o(*args),o.eval(*args)) for o in op for args in product(ns, repeat = o.n_args) if 0<=o.eval(*args)<=n_max]
#print([(str(o),e) for o,e in op])
#opeq = [o==var for o,e in op]
#op = [(o(*args),o.eval(*args)) for o in opeq for args in product(ns, repeat = o.n_args)]
#print([(str(o),e) for o,e in opeq])

# -------------------------------
# Paths
# -------------------------------

data_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}/stimuli'
os.makedirs(f'{data_path}', exist_ok=True)
font_path = f'{os.environ.get("DATA_PATH")}/font'

# -------------------------------
# Drawing
# -------------------------------

bg_color = (0,0,0)
char_color = (255,255,255)

# Gen from MNIST
mnist = MNIST(f'{os.environ.get("NN_COMMON")}/data/dataset', download=True, transform=transforms.ToTensor(), train=True)
num = np.array([mnist[i][1] for i in range(len(mnist))])
char_dataset = {str(i): torch.utils.data.Subset(mnist, np.where(num == i)[0]) for i in range(0,n_max+1)}
gen_char = lambda c, char_dim, bg_color, char_color: gen_from_mnist_clean(char_dataset, c, char_dim, bg_color, char_color)

"""
# Gen from font
font_ext = {
    "FreeMono": ["", "Bold", "Oblique", "BoldOblique"],
    "Sono": ["-SemiBold", "-Bold", "-ExtraBold", "-ExtraLight", "-Light", "-Medium", "-Regular"]
}
font_file = [f'{font_path}/{k}{s}.ttf' for k,v in font_ext.items() for s in v]
font = [ft.Face(f) for f in font_file]
gen_char = lambda c, char_dim, bg_color, char_color: gen_from_font(font, c, char_dim, bg_color, char_color)
"""

# -------------------------------
# Generation loop
# -------------------------------

for (o,r) in tqdm(op):
    for i in range(n_sample):
        tensor = gen_tensor(o, gen_char, char_dim, bg_color, char_color)
        img = tensor_to_img(tensor)
        img.save(f'{data_path}/{r}_{str(o).replace(" ", "")}_{i}.png')
