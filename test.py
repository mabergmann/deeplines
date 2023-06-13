import argparse

import pytorch_lightning as pl
import pathlib
from pytorch_lightning.loggers import MLFlowLogger

from deeplines.datamodel import RandomDataModel
from deeplines.engine import Engine


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Deeplines test",
        description="Tests the Deeplines model",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of the image"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Width of the image"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Name of the dataset",
        choices=["random"]
    )
    parser.add_argument(
        "--n_columns",
        type=int,
        help="Number of columns outputed by the model"
    )
    parser.add_argument(
        "--anchors_per_column",
        type=int,
        help="Number of anchors in each column outputed by the model"
    )
    parser.add
    parser.add_argument(
        "--backbone",
        type=str,
        help="Backbone that should be used",
        choices=["resnet50", "vgg16"]
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size"
    )
    return parser.parse_args()


def test(args, run_id, experiment_id="524490727776501304"):
    pl.seed_everything(42, workers=True)

    data = RandomDataModel(args.batch_size, args.width, args.height)

    engine = Engine(args)

    logger = MLFlowLogger(
        experiment_name="deeplines",
        log_model=True,
        run_id=run_id
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger
    )

    ckpt_path = str(next(pathlib.Path(f"mlruns/{experiment_id}/{run_id}/checkpoints").glob("*.ckpt")))
    test_results = trainer.test(
        engine, data,
        ckpt_path=ckpt_path
    )

    print(test_results)


if __name__ == "__main__":
    args = parse_args()
    test(args)
