import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets.randomlines import RandomLines


class RandomDataModel(pl.LightningDataModule):
    def __init__(self, batch_size, width, height):
        super().__init__()
        self.batch_size = batch_size

        self.train_dataset = RandomLines(
            image_size=(width, height),
            min_lines=1,
            max_lines=1
        )
        self.val_dataset = RandomLines(
            image_size=(width, height),
            min_lines=1,
            max_lines=1
        )

        self.test_dataset = RandomLines(
            image_size=(width, height),
            min_lines=1,
            max_lines=1
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.batch_size),
            collate_fn=self.train_dataset.collate_fn,
            num_workers=0,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.batch_size),
            collate_fn=self.val_dataset.collate_fn,
            num_workers=12,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=12
        )
