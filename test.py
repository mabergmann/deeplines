import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from deeplines.datasets.randomlines import RandomLines
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
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    test_dataset = RandomLines(
        image_size=(args.width, args.height),
        min_lines=1,
        max_lines=5
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        num_workers=12
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    logger = MLFlowLogger(
        experiment_name="deeplines"
    )

    engine = Engine(
        n_columns=args.n_columns,
        image_size=(args.width, args.height),
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    test_results = trainer.test(
        engine,
        dataloaders=[test_loader],
        ckpt_path="mlruns/277891689813798793/038648790b024e8380e27f496cc2b7c9/checkpoints/epoch=10-step=5500.ckpt"
    )
    print(test_results)


if __name__ == "__main__":
    main()
