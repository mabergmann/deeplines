import argparse

from test import test
from train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='Deeplines train',
        description='Trains the Deeplines model',
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Value between 0 and 1, defining learning rate',
    )
    parser.add_argument(
        '--height',
        type=int,
        help='Height of the image',
    )
    parser.add_argument(
        '--width',
        type=int,
        help='Width of the image',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        help='Value between 0 and 1, defining weight decay',
    )
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        help='Name of the dataset',
        choices=['random', 'NKL'],
    )
    parser.add_argument(
        '--n_columns',
        type=int,
        help='Number of columns outputted by the model',
    )
    parser.add_argument(
        '--anchors_per_column',
        type=int,
        help='Number of anchors in each column outputted by the model',
    )
    parser.add_argument(
        '--regression_weight',
        type=float,
        help='Weight for the regression component of the loss function',
    )
    parser.add_argument(
        '--objectness_weight',
        type=float,
        help='Weight for the objectness component of the loss function',
    )
    parser.add_argument(
        '--no_objectness_weight',
        type=float,
        help='Weight for the no_objectness component of the loss function',
    )
    parser.add_argument(
        '--backbone',
        type=str,
        help='Backbone that should be used',
        choices=['resnet50', 'vgg16', 'pit_b', 'pit_ti'],
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
    )
    parser.add_argument(
        '--loss',
        type=str,
        help='Loss that should be used',
        choices=['hausdorff', 'MSE'],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = train(args)
    test(args, run_id)


if __name__ == '__main__':
    main()
