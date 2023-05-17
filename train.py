import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from deeplines.datamodel import RandomDataModel
from deeplines.engine import Engine


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Deeplines train",
        description="Trains the Deeplines model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Value between 0 and 1, defining lerarning rate"
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
        "--weight_decay",
        type=float,
        help="Value between 0 and 1, defining weight decay"
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


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    data = RandomDataModel(args.batch_size, args.width, args.height)

    engine = Engine(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    logger = MLFlowLogger(
        experiment_name="deeplines",
        log_model=True,
    )

    for k, v in vars(args).items():
        logger.experiment.log_param(logger.run_id, k, v)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        num_sanity_val_steps=0,
        max_epochs=500,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm"
    )
    trainer.tune(engine, datamodule=data)
    trainer.fit(engine, datamodule=data)


if __name__ == "__main__":
    main()
