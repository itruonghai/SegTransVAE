from data.brats import get_train_dataloader, get_val_dataloader, get_test_dataloader
import pytorch_lightning as pl
from trainer import BRATS
import os 
import torch
os.system('cls||clear')
print("Testing ...")

CKPT = ''
model = BRATS(use_VAE=True).load_from_checkpoint(CKPT).eval()
val_dataloader = get_val_dataloader()
test_dataloader = get_test_dataloader()
trainer = pl.Trainer(gpus = [0], precision=32, progress_bar_refresh_rate=10)

trainer.test(model, dataloaders = val_dataloader)
trainer.test(model, dataloaders = test_dataloader)

