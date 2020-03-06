import logging
from pathlib import Path

import click

from paq2piq.common import set_up_seed
from paq2piq.inference_model import InferenceModel, RoIPoolModel


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


def main():
    init_logging()
    cli.add_command(get_image_score)
    cli()


if __name__ == "__main__":
    main()
