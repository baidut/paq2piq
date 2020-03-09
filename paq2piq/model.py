import torch
import torch.nn as nn
import torchvision as tv
from torchvision.ops import RoIPool, RoIAlign
import numpy as np

"""
# %%
from paq2piq.model import *; RoIPoolModel()
# %%
"""

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def get_idx(batch_size, n_output, device=None):
    idx = torch.arange(float(batch_size), dtype=torch.float, device=device).view(1, -1)
    idx = idx.repeat(n_output, 1, ).t()
    idx = idx.contiguous().view(-1, 1)
    return idx

def get_blockwise_rois(blk_size, img_size=None):
    if img_size is None: img_size = [1, 1]
    y = np.linspace(0, img_size[0], num=blk_size[0] + 1)
    x = np.linspace(0, img_size[1], num=blk_size[1] + 1)
    a = []
    for n in range(len(y) - 1):
        for m in range(len(x) - 1):
            a += [x[m], y[n], x[m + 1], y[n + 1]]
    return a

class RoIPoolModel(nn.Module):
    rois = None
    criterion = nn.MSELoss()

    def __init__(self, backbone='resnet18'):
        super().__init__()
        if backbone is 'resnet18':
            model = tv.models.resnet18(pretrained=True)
            cut = -2
            spatial_scale = 1/32

        self.model_type = self.__class__.__name__
        self.body = nn.Sequential(*list(model.children())[:cut])
        self.head = nn.Sequential(
          AdaptiveConcatPool2d(),
          nn.Flatten(),
          nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.25, inplace=False),
          nn.Linear(in_features=1024, out_features=512, bias=True),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=512, out_features=1, bias=True)
        )
        self.roi_pool = RoIPool((2,2), spatial_scale)

    def forward(self, x):
        # compatitble with fastai model
        if isinstance(x, list) or isinstance(x, tuple):
            im_data, self.rois = x
        else:
            im_data = x

        feats = self.body(im_data)
        batch_size = im_data.size(0)

        if self.rois is not None:
            rois_data = self.rois.view(-1, 4)
            n_output = int(rois_data.size(0) / batch_size)
            idx = get_idx(batch_size, n_output, im_data.device)
            indexed_rois = torch.cat((idx, rois_data), 1)
            feats = self.roi_pool(feats, indexed_rois)
        preds = self.head(feats)
        return preds.view(batch_size, -1)

    def input_block_rois(self, blk_size=None, img_size=None, batch_size=1, include_image=True, device=None):
        if img_size is None:
            img_size = [1, 1]
        if blk_size is None:
            blk_size = [30, 30]

        a = [0, 0, img_size[1], img_size[0]] if include_image else []
        a += get_blockwise_rois(blk_size, img_size)
        t = torch.tensor(a).float().to(device)
        self.rois = t.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, -1).view(-1, 4)
