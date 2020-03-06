import numpy as np
import torch
from torchvision import transforms
from scipy import stats

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class Transform:
    def __init__(self):
        # normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

        self._train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self._val_transform = transforms.Compose([transforms.ToTensor()])

    @property
    def train_transform(self):
        return self._train_transform

    @property
    def val_transform(self):
        return self._val_transform

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


def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def format_output(global_score, local_scores=None):
    if local_scores is None:
        return {"global_score": float(global_score)}
    else:
        return {"global_score": float(global_score), "local_scores": local_scores}
