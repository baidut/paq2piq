from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

"""
# %%
from paq2piq.dataset import *
from paq2piq.common import Transform

transform = Transform()
p = Path('!data/FLIVE')
data = FLIVE(path_to_csv=p/'labels<=640_padded.csv',
             images_path=p/'<=640_padded',
             transform = transform.train_transform
             )
data.__getitem__(1)
# %%
"""


class FLIVE(Dataset):
    def __init__(self, path_to_csv: Path, images_path: Path, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
        row = self.df.iloc[item]

        image_path = self.images_path / row["name_image"]
        image = default_loader(image_path)
        x = self.transform(image)

        mos_image = row['mos_image']

        return x, mos_image
