from pathlib import Path

import torch
# from PIL.Image import Image
from PIL import Image, ImageSequence
from torchvision.datasets.folder import default_loader

from .common import Transform, format_output, render_output
from .model import *

"""
#######################
# %% show quality map
#######################
%matplotlib inline
from paq2piq.inference_model import *;

file = '/media/zq/Seagate/Git/fastiqa/images/Picture1.jpg'
model = InferenceModel(RoIPoolModel(), 'paq2piq/RoIPoolModel.pth')
image = Image.open(file)
output = model.predict_from_pil_image(image)
render_output(image, output)
# %%
"""

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class InferenceModel:
    blk_size = [20, 20]

    def __init__(self, model, path_to_model_state: Path):
        self.transform = Transform().val_transform
        model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
        self.model = model
        self.model.load_state_dict(model_state["model"])
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_file(self, image_path: Path, render=False):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image: Image):
        image = image.convert("RGB")
        return self.predict(image)

    def predict_from_vid_file(self, vid_path: Path):
        im = Image.open(vid_path)
        index = 1
        for frame in ImageSequence.Iterator(im):
            frame.save("frame%d.png" % index)
            index = index + 1
        pass

    @torch.no_grad()
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(device)
        self.model.input_block_rois(self.blk_size, [image.shape[-2], image.shape[-1]], device=device)
        t = self.model(image).data.cpu().numpy()[0]

        local_scores = np.reshape(t[1:], self.blk_size)
        global_score = t[0]
        return format_output(global_score, local_scores)
