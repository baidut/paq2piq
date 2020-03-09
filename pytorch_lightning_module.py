from paq2piq.dataset import FLIVE
from paq2piq.metric import IQAMetric
from paq2piq.model import RoIPoolModel
import pytorch_lightning as pl
from pathlib import Path

import torch # torch.optim.Adam
import torch.nn as nn # nn.MSELoss()
from torchvision import transforms
from torch.utils.data import DataLoader

class RoIPoolLightningModule(pl.LightningModule):
    path_to_save_csv = Path('!data/FLIVE/release')
    path_to_images = Path("!data/FLIVE/release/images")
    loss_func = nn.MSELoss()
    metrics = IQAMetric()

    def __init__(self, backbone='resnet18'):
        super().__init__()
        self.model = RoIPoolModel(backbone)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        self.metrics.update(y, y_hat)
        return {'val_loss': self.loss_func(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        lcc, srcc = self.metrics.lcc, self.metrics.srcc
        self.metrics.reset()
        tensorboard_logs = {'val_loss': avg_loss, 'lcc': lcc, 'srcc': srcc}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        self.metrics.update(y, y_hat)
        return {'test_loss': self.loss_func(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        lcc, srcc = self.metrics.lcc, self.metrics.srcc
        self.metrics.reset()
        logs = {'test_loss': avg_loss, 'lcc': lcc, 'srcc': srcc}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.model.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        train_ds = FLIVE(self.path_to_save_csv / "train.csv", self.path_to_images, transform=transforms.ToTensor())
        return DataLoader(train_ds, batch_size=64, shuffle=True)

    def val_dataloader(self):
        # OPTIONAL
        val_ds = FLIVE(self.path_to_save_csv / "val.csv", self.path_to_images, transform=transforms.ToTensor())
        return DataLoader(val_ds, batch_size=32, shuffle=False)

    def test_dataloader(self):
        # OPTIONAL
        test_ds = FLIVE(self.path_to_save_csv / "test.csv", self.path_to_images, transform=transforms.ToTensor())
        return DataLoader(test_ds, batch_size=1, shuffle=False)
