import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from deeplines.datasets.randomlines import RandomLines
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
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    train_dataset = RandomLines(
        image_size=(args.width, args.height),
        min_lines=1,
        max_lines=5
    )
    val_dataset = RandomLines(
        image_size=(args.width, args.height),
        min_lines=1,
        max_lines=5
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=train_dataset.collate_fn,
        num_workers=12
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.collate_fn,
        num_workers=12
    )

    engine = Engine(
        image_size=(args.width, args.height),
        n_columns=args.n_columns,
        lr=args.lr
    )

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
        logger=logger
    )
    trainer.fit(engine, train_loader, val_loader)


if __name__ == "__main__":
    main()
