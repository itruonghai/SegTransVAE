from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from trainer import BRATS
import os 
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
args = parser.parse_args()

os.system('cls||clear')

model = BRATS(use_VAE = True)
checkpoint_callback = ModelCheckpoint(
    monitor='val/MeanDiceScore',
    dirpath='./ckpt/{}'.format(args.exp),
    filename='Epoch{epoch:3d}-MeanDiceScore{val/MeanDiceScore:.4f}',
    save_top_k=3,
    mode='max',
    save_last= True,
    auto_insert_metric_name=False
)
early_stop_callback = EarlyStopping(
   monitor='val/MeanDiceScore',
   min_delta=0.0001,
   patience=15,
   verbose=False,
   mode='max'
)
tensorboardlogger = TensorBoardLogger(
    'logs', 
    name = args.exp, 
    default_hp_metric = None 
)
trainer = pl.Trainer(#fast_dev_run = 10, 
#                     accelerator='ddp',
                    #overfit_batches=5,
                     gpus = [0], 
                        precision=16,
                     max_epochs = 200, 
                     progress_bar_refresh_rate=10,  
                     callbacks=[checkpoint_callback, early_stop_callback], 
#                     auto_lr_find=True,
                    num_sanity_val_steps=2,
                    logger = tensorboardlogger,
#                     limit_train_batches=0.01, 
#                     limit_val_batches=0.01
                     )
# trainer.tune(model)
trainer.fit(model)

