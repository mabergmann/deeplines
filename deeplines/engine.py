import pytorch_lightning as pl
import torch

from .model import DeepLines
from .loss import DeepLineLoss


class Engine(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DeepLines((800, 800), 9)
        self.loss = DeepLineLoss((800, 800), 9)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log('val_loss', loss)
