from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from trainer import BRATS
import os 
os.system('cls||clear')

model = BRATS(use_VAE = True)
checkpoint_callback = ModelCheckpoint(
    monitor='val/MeanDiceScore',
    dirpath='./ckpt',
    filename='segtransvae_epoch{epoch:3d}_MeanDiceScore{val/MeanDiceScore:.4f}',
    save_top_k=3,
    mode='max',
    save_last= True,
    auto_insert_metric_name=False
)
trainer = pl.Trainer(#fast_dev_run = 10, 
#                     accelerator='ddp',
#                 overfit_batches=5,
                     gpus = [0], 
                        precision=16,
                     max_epochs = 200, 
                     progress_bar_refresh_rate=10,  
                     callbacks=[checkpoint_callback], 
#                     auto_lr_find=True,
                    num_sanity_val_steps=0,
#                     limit_train_batches=0.01, 
#                     limit_val_batches=0.01
                     )
trainer.logger._log_graph = True         # If True, we plot the computation graph 
trainer.logger._default_hp_metric = None 
# trainer.tune(model)
trainer.fit(model)

