import argparse
import pytorch_lightning as pl
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    engine = Engine(args)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
    )
    test_results = trainer.test(
        engine, data,
        ckpt_path="/home/matheus/Workspace/deeplines/mlruns/705148703057156592/1771c49392dc427bbd7de5be4c7e02fd/checkpoints/epoch=457-step=7328.ckpt"
    )

    print(test_results)


if __name__ == "__main__":
    main()
