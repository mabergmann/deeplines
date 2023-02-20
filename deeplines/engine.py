import pytorch_lightning as pl
import torch

from .loss import DeepLineLoss
from .metrics import MetricAccumulator
from .model import DeepLines
from . import utils


class Engine(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DeepLines((224, 224), 9)
        self.loss = DeepLineLoss((224, 224), 9)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        metric_accumulator = MetricAccumulator()
        x, y = val_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        lines = utils.get_lines_from_output(pred, 224, 224)
        metric_accumulator.update(lines, y)
        self.log('val_loss', loss)
        self.log('val_precision', metric_accumulator.get_precision())
        self.log('val_recall', metric_accumulator.get_recall())
        self.log('val_f1', metric_accumulator.get_f1())
        return loss

    def test_step(self, test_batch, batch_idx):
        metric_accumulator = MetricAccumulator()
        x, y = test_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        lines = utils.get_lines_from_output(pred, 224, 224)
        metric_accumulator.update(lines, y)
        self.log('test_loss', loss)
        self.log('test_precision', metric_accumulator.get_precision())
        self.log('test_recall', metric_accumulator.get_recall())
        self.log('test_f1', metric_accumulator.get_f1())
        return loss
