import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deeplines.datasets.randomlines import RandomLines
from deeplines.engine import Engine
from deeplines import utils

pl.seed_everything(42, workers=True)

train_dataset = RandomLines(image_size=(224, 224), min_lines=1, max_lines=5)
val_dataset = RandomLines(image_size=(224, 224), min_lines=1, max_lines=5)

train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn, num_workers=12)

engine = Engine()

trainer = pl.Trainer(accelerator="gpu", devices=1)
trainer.fit(engine, train_loader, val_loader)
