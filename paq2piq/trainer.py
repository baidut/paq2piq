import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .common import Transform
from .metric import IQAMetric
from .dataset import FLIVE
from .model import RoIPoolModel
from .DiscriminativeLR import discriminative_lr_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

"""
# %% images smaller than 640x640 are padded
# %% images bigger than 640x640 are not processed

#########################
# %% test get_dataloaders
#########################
from paq2piq.trainer import *
p = Path('!data/FLIVE/release')
get_dataloaders(path_to_save_csv=p,
                path_to_images=p/'images',
                batch_size=64,
                num_workers=16,
                )
#########################
# %% test validate_and_test
#########################

logging.basicConfig(level=logging.INFO)
from paq2piq.trainer import *
p = Path('!data/FLIVE/release')
validate_and_test(path_to_save_csv=p,
                path_to_images=p/'images',
                batch_size=64,
                num_workers=16,
                path_to_model_state='paq2piq/RoIPoolModel.pth'
                )
# %%
"""


def get_dataloaders(
    path_to_save_csv: Path, path_to_images: Path, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = Transform()

    train_ds = FLIVE(path_to_save_csv / "train.csv", path_to_images, transform.train_transform)
    val_ds = FLIVE(path_to_save_csv / "val.csv", path_to_images, transform.val_transform)
    test_ds = FLIVE(path_to_save_csv / "test.csv", path_to_images, transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # sizes are different, set batch size to 1
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader


def validate_and_test(
    path_to_save_csv: Path,
    path_to_images: Path,
    batch_size: int,
    num_workers: int,
    path_to_model_state: Path,
) -> None:
    _, val_loader, test_loader = get_dataloaders(
        path_to_save_csv=path_to_save_csv, path_to_images=path_to_images, batch_size=batch_size, num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoIPoolModel().to(device)

    criterion = model.criterion.to(device)
    best_state = torch.load(path_to_model_state)
    model.load_state_dict(best_state["model"])

    model.eval()
    val_metrics = IQAMetric()
    test_metrics = IQAMetric()
    val_loss = 0
    test_loss = 0

    with torch.no_grad():
        for (x, y) in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            val_loss += criterion(y_pred, y)
            val_metrics.update(y, y_pred)


    with torch.no_grad():
        for (x, y) in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_loss += criterion(y_pred, y)
            test_metrics.update(y, y_pred)

    val_loss /= len(val_loader.dataset)
    test_loss /= len(test_loader.dataset)

    logger.info(f"val loss={val_loss}, SRCC={val_metrics.srcc}, LCC={val_metrics.lcc};"
                f"test loss={test_loss}, SRCC={test_metrics.srcc}, LCC={test_metrics.lcc};")


def get_optimizer(optimizer_type: str, model: RoIPoolModel, init_lr: float) -> torch.optim.Optimizer:
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.5, weight_decay=9)
    else:
        raise ValueError(f"not such optimizer {optimizer_type}")
    return optimizer


class Trainer:
    def __init__(
        self,
        *,
        path_to_save_csv: Path,
        path_to_images: Path,
        num_epoch: int,
        model_type: str,
        num_workers: int,
        batch_size: int,
        init_lr: float,
        experiment_dir: Path,
        # drop_out: float,
        optimizer_type: str,
    ):

        train_loader, val_loader, _ = get_dataloaders(
            path_to_save_csv=path_to_save_csv,
            path_to_images=path_to_images,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = RoIPoolModel().to(self.device)
        params, lr_arr, _ = discriminative_lr_params(model, slice(1e-5, 1e-3))
        optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-1)
        # get_optimizer(optimizer_type=optimizer_type, model=model, init_lr=init_lr)

        self.model = model
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=list(lr_arr), max_lr=list(lr_arr*100))
        #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode="min", patience=5)
        self.criterion = model.criterion.to(self.device)
        self.model_type = model_type

        experiment_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(str(experiment_dir / "logs"))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0
        self.print_freq = 100

    def train_model(self):
        best_loss = float("inf")
        best_state = None
        for e in range(1, self.num_epoch + 1):
            train_loss = self.train()
            val_loss = self.validate()
            self.scheduler.step()
            #self.scheduler.step(metrics=val_loss) # for lr_scheduler.ReduceLROnPlateau

            self.writer.add_scalar("train/loss", train_loss, global_step=e)
            self.writer.add_scalar("val/loss", val_loss, global_step=e)

            if best_state is None or val_loss < best_loss:
                logger.info(f"updated loss from {best_loss} to {val_loss}")
                best_loss = val_loss
                best_state = {
                    "model": self.model.state_dict(), # compatible with fastai model
                    "model_type": self.model_type,
                    "epoch": e,
                    "best_loss": best_loss,
                }
                torch.save(best_state, self.experiment_dir / "best_state.pth")

    def train(self):
        self.model.train()
        train_loss = 0
        total_iter = len(self.train_loader.dataset) // self.train_loader.batch_size

        for idx, (x, y) in enumerate(self.train_loader):
            s = time.monotonic()

            x = x.to(self.device)
            y = y.to(self.device)


            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss #loss.item()

            self.writer.add_scalar("train/current_loss", loss, self.global_train_step)
            #self.writer.add_scalar("train/avg_loss", train_loss/(idx+1), self.global_train_step)
            self.global_train_step += 1

            e = time.monotonic()
            if idx % self.print_freq:
                log_time = self.print_freq * (e - s)
                eta = ((total_iter - idx) * log_time) / 60.0
                print(f"iter #[{idx}/{total_iter}] " f"loss = {loss:.3f} " f"time = {log_time:.2f} " f"eta = {eta:.2f}")

        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_metrics = IQAMetric()

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                val_loss += loss
                self.writer.add_scalar("val/current_loss", loss, self.global_val_step)
                #self.writer.add_scalar("val/avg_loss", val_loss/(idx+1), self.global_val_step)
                val_metrics.update(y, y_pred)

                self.global_val_step += 1

        val_loss /= len(self.val_loader.dataset)
        self.writer.add_scalar("val/srcc", val_metrics.srcc, self.global_val_step)
        self.writer.add_scalar("val/lcc", val_metrics.lcc, self.global_val_step)
        return val_loss
