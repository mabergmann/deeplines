import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets.nkl import NKL
from .datasets.randomlines import RandomLines


class RandomDataModel(pl.LightningDataModule):
    def __init__(self, batch_size: int, width: int, height: int, dataset="NKL") -> None:
        super().__init__()
        self.batch_size = batch_size

        if dataset == "NKL":
            self.train_dataset = NKL(
                image_size=(width, height),
                split_file="data/train.txt",
                images_folder="data"
            )
            self.val_dataset = NKL(
                image_size=(width, height),
                split_file="data/val.txt",
                images_folder="data"
            )

            self.test_dataset = NKL(
                image_size=(width, height),
                split_file="data/val.txt",
                images_folder="data"
            )
        elif dataset == "random":
            self.train_dataset = RandomLines(
                image_size=(width, height),
                min_lines=1,
                max_lines=5
            )
            self.val_dataset = RandomLines(
                image_size=(width, height),
                min_lines=1,
                max_lines=5
            )

            self.test_dataset = RandomLines(
                image_size=(width, height),
                min_lines=1,
                max_lines=5
            )
        else:
            raise ValueError(f"{dataset} is not a valid option for dataset")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.batch_size),
            collate_fn=self.train_dataset.collate_fn,
            num_workers=12,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.batch_size),
            collate_fn=self.val_dataset.collate_fn,
            num_workers=12,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=12,
        )
