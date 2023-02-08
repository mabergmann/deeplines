import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deeplines.datasets.randomlines import RandomLines
from deeplines.engine import Engine

train_dataset = RandomLines(image_size=(800, 800), min_lines=1, max_lines=5)
val_dataset = RandomLines(image_size=(800, 800), min_lines=1, max_lines=5)

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

engine = Engine()

trainer = pl.Trainer(gpus=1, precision=32, limit_train_batches=0.5)
trainer.fit(engine, train_loader, val_loader)
