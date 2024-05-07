import torch
import torch.nn as nn
from cornet import cornet_s
from cornet.cornet_s import Flatten, Identity, CORblock_S
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import functional as FM
import os
from collections import OrderedDict

class ExtendedCORnet(pl.LightningModule):
    def __init__(self, out_features, pretrained = True, loss = nn.CrossEntropyLoss(), optimizer = 'adam',  lr = 1e-3, map_location=None):
        super(ExtendedCORnet, self).__init__()
        self.model = cornet_s(pretrained = pretrained, map_location = map_location)
        if out_features > 0:
            self.model.decoder.linear = nn.Linear(512, out_features)
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        return loss, acc

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            elif self.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
            else:
                raise NameError(f'Unknown optimizer {self.optimizer}')
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

class ScaledReLU(nn.Module):
    def __init__(self, scale, inplace = False):
        super(ScaledReLU, self).__init__()
        self.inplace = inplace
        self.scale = scale
    
    def forward(self, x):
        return F.relu(x*self.scale, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return f'shift={self.scale}{inplace_str}'

class ScaledCORnet(ExtendedCORnet):
    def __init__(self, scale = 1.0, times = None, modules = ['V1', 'V2', 'V4', 'IT'], fixed_batchnorm = False, **kwargs):
        super(ScaledCORnet, self).__init__(**kwargs)
        if 'V1' in modules:
            for i in range(1,3):
                setattr(self.model.V1, f'nonlin{i}', ScaledReLU(scale=scale, inplace=True))
        for j,(name,module) in enumerate(zip(['V2', 'V4', 'IT'], [self.model.V2, self.model.V4, self.model.IT])):
            if name in modules:
                if not times is None:
                    module.__init__(module.conv_input.in_channels, module.conv_input.out_channels, times[j])
                for i in range(1,4):
                    setattr(module, f'nonlin{i}', ScaledReLU(scale=scale, inplace=True))
        if fixed_batchnorm:
            out_channels = self.model.V1.conv1.out_channels
            self.model.V1.norm1 = nn.BatchNorm2d(out_channels, affine=False)
            self.model.V1.norm2 = nn.BatchNorm2d(out_channels, affine=False)
            for module in ['V2', 'V4', 'IT']:
                m = getattr(self.model, module)
                out_channels = m.conv_input.out_channels
                m.norm_skip = nn.BatchNorm2d(out_channels, affine=False)
                for t in range(m.times):
                    setattr(m, f'norm1_{t}', nn.BatchNorm2d(out_channels * m.scale, affine=False))
                    setattr(m, f'norm2_{t}', nn.BatchNorm2d(out_channels * m.scale, affine=False))
                    setattr(m, f'norm3_{t}', nn.BatchNorm2d(out_channels, affine=False))

class ZeroMask(nn.Module):
    def __init__(self, mask):
        super(ZeroMask, self).__init__()
        self.mask = mask
    
    def forward(self, x):
        x[:,self.mask] = 0
        return x

class ShuffleMask(nn.Module):
    def __init__(self, mask):
        super(ShuffleMask, self).__init__()
        self.mask = mask
    
    def forward(self, x):
        y = x[torch.randperm(x.shape[0])]
        x[:,self.mask] = y[:,self.mask]
        return x