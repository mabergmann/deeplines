import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deeplines.datasets.randomlines import RandomLines
from deeplines.engine import Engine
from deeplines import utils

pl.seed_everything(42, workers=True)

test_dataset = RandomLines(image_size=(224, 224), min_lines=1, max_lines=5)

test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn, num_workers=12)

engine = Engine()

trainer = pl.Trainer(accelerator="gpu", devices=1)
test_results = trainer.test(engine, dataloaders=[test_loader], ckpt_path="/home/matheus/Workspace/deeplines/lightning_logs/version_6/checkpoints/epoch=999-step=500000.ckpt")
print(test_results)
