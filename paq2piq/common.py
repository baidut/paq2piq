import numpy as np
import torch
from torchvision import transforms


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
