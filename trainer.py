# Torch Library
import torch
import torch.nn.functional as F
import torch.nn as nn 

# MONAI
from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Activations, Compose

# Pytorch Lightning
import pytorch_lightning as pl

# Custom Libraries
from models.SegTransVAE import SegTransVAE
from data.brats import get_train_dataloader, get_val_dataloader
from models.VAE import loss_vae

class BRATS(pl.LightningModule):
    def __init__(self, use_VAE = False, lr = 1e-4):
        super().__init__()
        # self.model = SegResNet(
        #         blocks_down = [1,2,2,4],
        #         blocks_up = [1,1,1],
        #         init_filters = 16,
        #         in_channels = 4,
        #         out_channels = 3, 
        #         dropout_prob = 0.2
        #         )
        self.use_vae = use_VAE
        self.lr = lr
        self.model = SegTransVAE(128, 8, 4, 3, 512, 8, 4, 4096, use_VAE = use_VAE)
        self.loss_vae = loss_vae()
        self.loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.post_trans_images = Compose(
                [Activations(sigmoid=True), 
                 AsDiscrete(threshold_values=True), 
                 ]
            )
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.best_val_dice = 0
        # self.example_input_array = torch.rand((1,4,128,128,128))
    def forward(self, x, is_validation = True):
        return self.model(x, is_validation) 
    def training_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        if not self.use_vae:
            outputs = self.forward(inputs, is_validation=False)
            loss = self.loss_function(outputs, labels)
        else:
            outputs, vae_out, mu, sigma = self.forward(inputs, is_validation=False)
            vae_loss = 0.1 * self.loss_vae(vae_out, inputs, mu, sigma)
            dice_loss = self.loss_function(outputs, labels)
            loss = vae_loss  +  dice_loss 
            self.log('train/vae_loss', vae_loss)
            self.log('train/dice_loss', dice_loss)
                
        self.log('train/loss', loss)
        return loss
    def validation_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        # print(inputs.shape, labels.shape)
    
        roi_size = (128, 128, 128)
        sw_batch_size = 1
        val_outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, self.model, overlap = 0.5)
        loss = self.loss_function(val_outputs, labels)
        val_outputs = [self.post_trans_images(i) for i in decollate_batch(val_outputs)]
        self.dice_metric(y_pred=val_outputs, y=labels)
        self.dice_metric_batch(y_pred=val_outputs, y=labels)
        return {"val_loss": loss, "val_number": len(val_outputs)}
    
    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        metric_batch = self.dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()
        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.log('val/MeanDiceScore', mean_val_dice)
        self.log('val/DiceTC', metric_tc)
        self.log('val/DiceWT', metric_wt)
        self.log('val/DiceET', metric_et)
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
                f"\n Current epoch: {self.current_epoch} Current mean dice: {mean_val_dice:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\n Best mean dice: {self.best_val_dice:.4f}"
                f" at epoch: {self.best_val_epoch}"
            )
        return {'val_MeanDiceScore': mean_val_dice}
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
#         optimizer = AdaBelief(self.model.parameters(), 
#                             lr=self.lr, eps=1e-12, 
#                             betas=(0.9,0.999), weight_decouple = True, 
#                             rectify = False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return get_train_dataloader()
    
    def val_dataloader(self):
        return get_val_dataloader()