import cv2
import pytorch_lightning as pl
import torch

from .loss import DeepLineLoss
from .metrics import MetricAccumulator
from .model import DeepLines
from . import utils


class Engine(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.image_size = (args.width, args.height)
        self.n_columns = args.n_columns
        self.args = args

        self.model = DeepLines(self.n_columns, args.backbone)
        self.loss = DeepLineLoss(self.image_size, self.n_columns)

        self.train_metric_accumulator = MetricAccumulator()
        self.val_metric_accumulator = MetricAccumulator()
        self.test_metric_accumulator = MetricAccumulator()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        lines = utils.get_lines_from_output(
            pred,
            self.image_size[0],
            self.image_size[1]
        )
        self.train_metric_accumulator.update(lines, y)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        self.log('train_precision', self.train_metric_accumulator.get_precision())
        self.log('train_recall', self.train_metric_accumulator.get_recall())
        self.log('train_f1', self.train_metric_accumulator.get_f1())
        self.train_metric_accumulator.reset()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        lines = utils.get_lines_from_output(
            pred,
            self.image_size[0],
            self.image_size[1]
        )
        self.val_metric_accumulator.update(lines, y)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        self.log('val_precision', self.val_metric_accumulator.get_precision())
        self.log('val_recall', self.val_metric_accumulator.get_recall())
        self.log('val_f1', self.val_metric_accumulator.get_f1())
        self.val_metric_accumulator.reset()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        lines = utils.get_lines_from_output(
            pred,
            self.image_size[0],
            self.image_size[1]
        )
        self.test_metric_accumulator.update(lines, y)
        self.log('test_loss', loss)

        images = utils.draw_result(x, lines)
        for n, i in enumerate(images):
            cv2.imwrite(f"output/{batch_idx}_{n}.png", i)


        return loss

    def on_test_epoch_end(self):
        self.log('test_precision', self.test_metric_accumulator.get_precision())
        self.log('test_recall', self.test_metric_accumulator.get_recall())
        self.log('test_f1', self.test_metric_accumulator.get_f1())
        self.test_metric_accumulator.reset()
