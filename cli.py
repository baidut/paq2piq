import logging
from pathlib import Path

import click

from paq2piq.common import set_up_seed
from paq2piq.inference_model import InferenceModel, RoIPoolModel
from paq2piq.trainer import Trainer, validate_and_test

def init_logging() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@click.group()
def cli():
    pass

"""
# %%
!ls paq2piq/*.py
!python /media/zq/Seagate/Git/paq2piq/cli.py --help
!python /media/zq/Seagate/Git/paq2piq/cli.py get-image-score --help
!python /media/zq/Seagate/Git/paq2piq/cli.py get-image-score --path_to_model_state paq2piq/RoIPoolModel.pth --path_to_image /media/zq/Seagate/Git/paq2piq/images/Picture1.jpg
#
#
pip install click
# %%
"""

@click.command("get-image-score", short_help="Get image scores")
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=str)
@click.option("--path_to_image", help="image ", required=True, type=str)
def get_image_score(path_to_model_state, path_to_image):
    model = InferenceModel(RoIPoolModel(), path_to_model_state=path_to_model_state)
    result = model.predict_from_file(path_to_image)
    click.echo(result)

"""
# %%
!python --version
!/home/zq/.virtualenvs/gpu/bin/python /media/zq/Seagate/Git/paq2piq/cli.py validate-model \
--path_to_model_state paq2piq/RoIPoolModel.pth \
--path_to_save_csv "!data/FLIVE/release" \
--path_to_images "!data/FLIVE/release/images" \
--batch_size 64
# %%
"""

@click.command("validate-model", short_help="Validate model")
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=Path)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True, type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=16, type=int)
# @click.option("--drop_out", help="drop out", default=0.0, type=float)
def validate_model(path_to_model_state, path_to_save_csv, path_to_images, batch_size, num_workers): # , drop_out
    validate_and_test(
        path_to_model_state=path_to_model_state,
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        batch_size=batch_size,
        num_workers=num_workers,
        # drop_out=drop_out,
    )
    click.echo("Done!")

"""
# %%
!/home/zq/.virtualenvs/gpu/bin/python
!python /media/zq/Seagate/Git/paq2piq/cli.py train-model \
--path_to_save_csv "!data/FLIVE/release" \
--path_to_images "!data/FLIVE/release/images" \
--experiment_dir "data/exp/t1-baseline" \
--num_epoch 100 \
--batch_size 64
# sh train.sh
# %%
"""

@click.command("train-model", short_help="Train model")
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", required=True, type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--experiment_dir", help="directory name to save all logs and weight", required=True, type=Path)
@click.option("--model_type", help="res net model type", default="resnet18", type=str)
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=16, type=int)
@click.option("--num_epoch", help="number of epoch", default=32, type=int)
@click.option("--init_lr", help="initial learning rate", default=0.0001, type=float)
# @click.option("--drop_out", help="drop out", default=0.5, type=float)
@click.option("--optimizer_type", help="optimizer type", default="adam", type=str)
@click.option("--seed", help="random seed", default=42, type=int)
def train_model(
    path_to_save_csv: Path,
    path_to_images: Path,
    experiment_dir: Path,
    model_type: str,
    batch_size: int,
    num_workers: int,
    num_epoch: int,
    init_lr: float,
    # drop_out: float,
    optimizer_type: str,
    seed: int,
):
    click.echo("Train and validate model")
    set_up_seed(seed)
    trainer = Trainer(
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        experiment_dir=experiment_dir,
        model_type=model_type,
        batch_size=batch_size,
        num_workers=num_workers,
        num_epoch=num_epoch,
        init_lr=init_lr,
        # drop_out=drop_out,
        optimizer_type=optimizer_type,
    )
    trainer.train_model()
    click.echo("Done!")


def main():
    init_logging()
    cli.add_command(get_image_score)
    cli.add_command(validate_model)
    cli.add_command(train_model)
    cli()


if __name__ == "__main__":
    main()
