import argparse
from train import train
from test import test


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
        "--anchors_per_column",
        type=int,
        help="Number of anchors in each column outputed by the model"
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
    run_id = train(args)
    test(args, run_id)


if __name__ == "__main__":
    main()
