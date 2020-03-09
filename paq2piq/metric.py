import torch
from scipy import stats
import numpy as np

class IQAMetric(object):
    idx_col = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.targs = []
        self.preds = []

    # only compute on image score
    def update(self, targs, preds):
        self.targs += targs
        self.preds += preds

    def compute(self, func):
        def get_column(x):
            x = torch.stack(x)
            if x.dim() != 1:
                x = x[:, self.idx_col].view(1, -1)[0]
            return np.asarray(x.cpu())
        return func(get_column(self.preds), get_column(self.targs))[0]

    @property
    def srcc(self):
        return self.compute(stats.spearmanr)

    @property
    def lcc(self):
        return self.compute(stats.pearsonr)

    @property
    def krocc(self):
        return self.compute(stats.kendalltau)
