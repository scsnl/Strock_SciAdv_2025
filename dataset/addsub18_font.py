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
from collections import OrderedDict

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

data_path = f'{os.environ.get("DATA_PATH")}/addsub_{n_max}_font/stimuli'
os.makedirs(f'{data_path}', exist_ok=True)
font_path = f'{os.environ.get("DATA_PATH")}/font'

# -------------------------------
# Drawing
# -------------------------------

bg_color = (0,0,0)
char_color = (255,255,255)

# Gen from font
font_ext = OrderedDict([
    ("FreeMono", ["", "Bold", "Oblique", "BoldOblique"]),
    ("Sono", ["-SemiBold", "-Bold", "-ExtraBold", "-ExtraLight", "-Light", "-Medium", "-Regular"]),
    ("Anonymous Pro", ["", " B", " I", " BI"]),
    ("VeraMono", ["", "-Bold", "-Italic", "-Bold-Italic"]),
    ("Cascadia", [""]),
    ("ComicMono", ["", "-Bold"]),
    ("CourierPrime", ["-Regular", "-Bold", "-Italic", "-BoldItalic"]),
    ("DejaVuSansMono", ["", "-Bold", "-Oblique", "-BoldOblique"]),
    ("UbuntuMono", ["-Regular", "-Bold", "-Italic", "-BoldItalic"]),
    ("RobotoMono", ["-Regular", "-Italic"] + [s+s2 for s in ["-SemiBold", "-Bold", "-ExtraLight", "-Light", "-Medium", "-Thin"] for s2 in ["", "Italic"]])
])
font_file = OrderedDict([(f'{k}{s}',f'{font_path}/{k}{s}.ttf') for k,v in font_ext.items() for s in v])
#for k, f in font_file.items():
#    print(k, f)
#    _f = ft.Face(f)
font = OrderedDict([(k,ft.Face(f)) for k,f in font_file.items()])
gen_char = lambda c, char_dim, bg_color, char_color: gen_from_font(font, c, char_dim, bg_color, char_color)

# -------------------------------
# Generation loop
# -------------------------------

for (o,r) in tqdm(op):
    for i in range(n_sample):
        tensor = gen_tensor(o, gen_char, char_dim, bg_color, char_color)
        img = tensor_to_img(tensor)
        img.save(f'{data_path}/{r}_{str(o).replace(" ", "")}_{i}.png')