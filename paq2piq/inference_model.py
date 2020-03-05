from pathlib import Path

import torch
from PIL.Image import Image
from torchvision.datasets.folder import default_loader

from paq2piq.common import Transform, format_output
from paq2piq.model import *

"""
# %%
from paq2piq.inference_model import *;

file = '/media/zq/Seagate/Git/fastiqa/images/Picture1.jpg'
model = InferenceModel(RoIPoolModel(), 'paq2piq/RoIPoolModel.pth')
model.predict_from_file(file)
# %%
"""

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class InferenceModel:
    def __init__(self, model, path_to_model_state: Path):
        self.transform = Transform().val_transform
        model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
        self.model = model
        self.model.load_state_dict(model_state["model"])
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_file(self, image_path: Path):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image: Image):
        image = image.convert("RGB")
        return self.predict(image)

    @torch.no_grad()
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(device)
        prob = self.model(image).data.cpu().numpy()[0]

        #mean_score = get_mean_score(prob)
        #std_score = get_std_score(prob)
        # qmat later
        return prob #score
