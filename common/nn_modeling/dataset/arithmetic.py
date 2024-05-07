import torch
from torchvision import datasets, transforms, utils
import os
from torch.utils.data import Dataset
from PIL import Image
import pytorch_lightning as pl
import numpy as np
from torch import from_numpy
from pathlib import Path
import pickle
from functools import reduce
import copy
from torchvision.transforms.functional import rotate, affine
import freetype as ft
import glob
import re
from pathlib import PurePath
from tqdm import tqdm

# -------------------------------
# Format choices
# -------------------------------


def int_fmt(n_digit=None):
    if n_digit is None:
        return lambda x: f'{{{x}}}' if type(x) is str else f'{{:}}'.format(x)
    else:
        return lambda x: f'{{{x}:{n_digit}}}' if type(x) is str else f'{{:{n_digit}}}'.format(x)

# -------------------------------
# Operation choices
# -------------------------------

class Operation():
    def __init__(self, n_args):
        self.n_args = n_args
        self.simplify()
    def __str__(self):
        var = [chr(ord('a')+k) for k in range(self.n_args)]
        return self.to_str(var)
    def __repr__(self):
        name = self.__class__.__name__
        param = f'({reduce(lambda s1,s2: f"{s1},{s2}", [repr(o) for o in self.op])})' if hasattr(self, "op") else ''
        return f'{name}{param}'
    def __call__(self, *args):
        if hasattr(self, "op"):
            op = []
            j = 0
            for i,o in enumerate(self.op):
                op.append(o(*args[j:j+o.n_args]))
                j+=o.n_args
            return self.__class__(op)            
        elif self.n_args == 0 or args[0] is None:
            op = self
        else:
            op = Constant(args[0], fmt=self.fmt)
        return op
    def __add__(self, op):
        return Add([self,op])
    def __mul__(self, op):
        return Mul([self,op])
    def __sub__(self, op):
        return Sub([self,op])
    def __eq__(self, op):
        return Equal([self,op])
    def eval(self, *args):
        kwargs = {chr(ord('a')+k):args[k] for k in range(self.n_args)}
        s = str(self).format(**kwargs)
        s = s.replace('x', '*')
        s = s.replace('=', '==')
        return eval(s)
    def simplify(self):
        pass

class Constant(Operation):
    def __init__(self, value, fmt = None):
        self.fmt = int_fmt() if fmt is None else fmt
        self.p = 0
        self.value = value
        super().__init__(0)
    def to_str(self, var):
        return self.fmt(self.value)
    def __repr__(self):
        return str(self.value)

class Var(Operation):
    def __init__(self, fmt = None):
        self.fmt = int_fmt() if fmt is None else fmt
        self.p = 0
        super().__init__(1)
    def to_str(self, var):
        return f'{self.fmt(var[0])}'

class Add(Operation):
    def __init__(self, op):
        l = []
        self.op = op
        self.p = 2
        super().__init__(sum(o.n_args for o in op))

    def to_str(self, var):
        idx = [0]+list(np.cumsum([o.n_args for o in self.op]))
        s = [f'{"(" if o.p > self.p else ""}{o.to_str(var[idx[i]:idx[i+1]])}{")" if o.p > self.p else ""}' for i,o in enumerate(self.op)]
        return reduce(lambda s1,s2: f'{s1}+{s2}', s)
    def simplify(self):
        l = []
        for o in self.op:
            if o.__class__ == Add:
                l = l + o.op
            else:
                l.append(o)
        self.op = l

class Mul(Operation):
    def __init__(self, op):
        self.op = op
        self.p = 1
        super().__init__(sum(o.n_args for o in op))
    def to_str(self, var):
        idx = [0]+list(np.cumsum([o.n_args for o in self.op]))
        s = [f'{"(" if o.p > self.p else ""}{o.to_str(var[idx[i]:idx[i+1]])}{")" if o.p > self.p else ""}' for i,o in enumerate(self.op)]
        return reduce(lambda s1,s2: f'{s1}x{s2}', s)
    def simplify(self):
        l = []
        for o in self.op:
            if o.__class__ == Mul:
                l = l + o.op
            else:
                l.append(o)
        self.op = l

class Sub(Operation):
    def __init__(self, op):
        self.op = op
        self.p = 2
        super().__init__(sum(o.n_args for o in op))
    def to_str(self, var):
        idx = [0]+list(np.cumsum([o.n_args for o in self.op]))
        s = [f'{"(" if o.p > self.p else ""}{o.to_str(var[idx[i]:idx[i+1]])}{")" if o.p > self.p else ""}' for i,o in enumerate(self.op)]
        return reduce(lambda s1,s2: f'{s1}-{s2}', s)
    def simplify(self):
        if self.op[0].__class__ == Sub:
            self.op = self.op[0].op + self.op[1:]

class Equal(Operation):
    def __init__(self, op):
        l = []
        self.op = op
        self.p = 3
        super().__init__(sum(o.n_args for o in self.op))
    def to_str(self, var):
        idx = [0]+list(np.cumsum([o.n_args for o in self.op]))
        s = [f'{"(" if o.p > self.p else ""}{o.to_str(var[idx[i]:idx[i+1]])}{")" if o.p > self.p else ""}' for i,o in enumerate(self.op)]
        return reduce(lambda s1,s2: f'{s1}={s2}', s)


# -------------------------------
# Character generation choices
# -------------------------------

def gen_from_mnist(dataset, c, char_dim, bg_color, char_color):
    if c == " ":
        t = torch.zeros((1,)+char_dim)
    elif c == "-":
        idx = np.random.randint(len(dataset['1']))
        t = dataset['1'][idx][0].transpose(1,2)
    elif c == "+":
        idx = np.random.randint(len(dataset['1']))
        t =  torch.clamp(dataset['1'][idx][0]+rotate(dataset['1'][idx][0], 90),0,1)
    elif c == "x":
        idx = np.random.randint(len(dataset['1']))
        t =  torch.clamp(rotate(dataset['1'][idx][0], 45)+rotate(dataset['1'][idx][0], 135),0,1)
    elif c == "=":
        idx = np.random.randint(len(dataset['1']))
        t =  torch.clamp(affine(dataset['1'][idx][0], 90, (0,char_dim[1]//4), 1, 0)+affine(dataset['1'][idx][0], 90, (0,-char_dim[1]//4), 1, 0),0,1)[0]
    else:
        idx = np.random.randint(len(dataset[c]))
        t = dataset[c][idx][0]
    return bg_color + t*(char_color-bg_color)

def check_vertical_one(img, threshold = 1.5):
    x = torch.sum(img, axis = (0,1))/torch.sum(img)
    mean = torch.sum(x*torch.arange(len(x)))
    std = torch.sqrt(torch.sum(x*(torch.arange(len(x))-mean)**2))
    return std<threshold

def random_vertical_one(dataset, threshold = 1.5):
    idx = np.random.randint(len(dataset['1']))
    img = dataset['1'][idx][0]
    while not check_vertical_one(img, threshold):
        idx = np.random.randint(len(dataset['1']))
        img = dataset['1'][idx][0]
    return idx

def gen_from_mnist_clean(dataset, c, char_dim, bg_color, char_color, threshold = 1.5):
    if c == " ":
        t = torch.zeros((1,)+char_dim)
    elif c == "-":
        idx = random_vertical_one(dataset, threshold)
        t = dataset['1'][idx][0].transpose(1,2)
    elif c == "+":
        idx = random_vertical_one(dataset, threshold)
        t =  torch.clamp(dataset['1'][idx][0]+rotate(dataset['1'][idx][0], 90),0,1)
    elif c == "x":
        idx = random_vertical_one(dataset, threshold)
        t =  torch.clamp(rotate(dataset['1'][idx][0], 45)+rotate(dataset['1'][idx][0], 135),0,1)
    elif c == "=":
        idx = random_vertical_one(dataset, threshold)
        t =  torch.clamp(affine(dataset['1'][idx][0], 90, (0,char_dim[1]//4), 1, 0)+affine(dataset['1'][idx][0], 90, (0,-char_dim[1]//4), 1, 0),0,1)[0]
    else:
        idx = np.random.randint(len(dataset[c]))
        t = dataset[c][idx][0]
    return bg_color + t*(char_color-bg_color)

no_transform = transforms.ToTensor()

def gen_from_font(font, c, char_dim, bg_color, char_color):
    t = torch.zeros(char_dim)
    if c != " ":
        idx = np.random.randint(len(font))
        face = font[idx]
        face.set_pixel_sizes(*char_dim)
        face.load_char(c)
        w,h = face.glyph.bitmap.width, face.glyph.bitmap.rows
        x,y = face.glyph.bitmap_left, (face.size.ascender >> 6) - face.glyph.bitmap_top - ((face.size.ascender >> 6)-(face.size.descender >> 6)-char_dim[0])//2
        t[y:y+h, x:x+w] = torch.Tensor(np.array(face.glyph.bitmap.buffer, dtype='ubyte').reshape(h,w)/255)
    return bg_color + t*(char_color-bg_color)

# -------------------------------
# Core generations of img
# -------------------------------

tensor_to_img = transforms.ToPILImage()

def gen_tensor(op, gen_char, char_dim, bg_color, char_color, space = 0):
    bg_color = no_transform(np.array(bg_color, dtype = np.uint8).reshape((1,1,3)))
    char_color = no_transform(np.array(char_color, dtype = np.uint8).reshape((1,1,3)))
    s = str(op)
    n_char = len(s)
    t = bg_color+torch.zeros((3,char_dim[1]+2*space,n_char*(char_dim[0]+space)+space))
    for i in range(n_char):
        c = gen_char(s[i], char_dim, bg_color, char_color)
        t[:,space:space+char_dim[1],i*(char_dim[0]+space)+space:(i+1)*(char_dim[0]+space)] = c
    return t

# -------------------------------
# Interface with pytorch
# -------------------------------

default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ArithmeticDataset(Dataset):

    def __init__(self, path, summary = None, include = None, transform = default_transform):
        if summary is None:
            summary = f'{path}/summary.pkl'
        if os.path.exists(summary):
            self.load_pkl(summary)
        else:
            self.load_png(path, include, transform)
            self.save_pkl(summary)

    def __len__(self):
        return len(self.label_id)

    def get_res_from_path(self,path):
        return int(PurePath(path).parts[-1].split('_')[0])

    def get_op_from_path(self,path):
        return PurePath(path).parts[-1].split('_')[1]

    def get_id_from_path(self,path):
        return int(PurePath(path).parts[-1].split('.')[0].split('_')[2])

    def load_png(self, path, include, transform):
        print("Loading PNG data")
        path = sorted(glob.glob(path))
        if not include is None:
            path = sorted([p for p in path if re.fullmatch(include, p)])
        self.all_label, self.label_id = np.unique([self.get_res_from_path(p) for p in path], return_inverse=True)
        self.all_op, self.op_id = np.unique([self.get_op_from_path(p) for p in path], return_inverse=True)
        self.img = []
        for p in tqdm(path):
            self.img.append(transform(Image.open(p)))
        self.img = torch.stack(self.img)

    def load_pkl(self, path):
        print("Loading PKL data")
        with open(path, 'rb') as f:
            pkl_data = pickle.load(f)
            for k,v in pkl_data.items():
                setattr(self, k, v)

    def save_pkl(self, path):
        print("Saving PKL data")
        with open(path, 'wb') as f:
            l = ['all_label', 'label_id', 'all_op', 'op_id', 'img']
            d = {k:getattr(self, k) for k in l}
            pickle.dump(d, f)

    def __getitem__(self, idx):
        img = self.img[idx]
        label_id = self.label_id[idx]
        op_id = self.op_id[idx]
        return  img,label_id,op_id