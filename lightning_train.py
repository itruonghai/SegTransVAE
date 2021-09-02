from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from trainer import BRATS
import os 
os.system('cls||clear')

model = BRATS()
checkpoint_callback = ModelCheckpoint(
    monitor='val_MeanDiceScore',
    dirpath='./ckpt',
    filename='segresnet-{epoch:3d}-{val dice score: .2f}',
    save_top_k=1,
    mode='max',
    save_last= True,
)
trainer = pl.Trainer(fast_dev_run = 10, 
                    # accelerator='ddp2',
#                 overfit_batches=5,
                     gpus = [0], 
                        precision=16,
                     max_epochs = 100, 
                     progress_bar_refresh_rate=10,  
                     callbacks=[checkpoint_callback], 
                     )
trainer.logger._log_graph = True         # If True, we plot the computation graph 
trainer.logger._default_hp_metric = None 
trainer.fit(model)
