from monai.networks.nets import SegResNet, UNETR
from monai.metrics import compute_meandice, DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import torch
from monai.transforms import AsDiscrete, Activations, Compose
import pytorch_lightning as pl
from models.TransBTS import TransformerBTS
from data.brats import get_train_dataloader, get_val_dataloader
import torch.nn.functional as F
import torch.nn as nn 
class loss_vae(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, sigma):
        mse = F.mse_loss(recon_x, x)
        kld = 0.5 * torch.mean(mu ** 2 + sigma ** 2 - torch.log(1e-8 + sigma ** 2) - 1)
        loss = mse + kld
        return loss
class BRATS(pl.LightningModule):
    def __init__(self, use_VAE = False):
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
        self.model = TransformerBTS(128, 8, 4, 3, 512, 8, 4, 4096, use_VAE = use_VAE)
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
#         self.example_input_array = torch.rand((1,4,128,128,128))
    def forward(self, x, is_validation = True):
        return self.model(x, is_validation) 
    def training_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        if not self.use_vae:
            outputs = self.forward(inputs, is_validation=False)
            loss = self.loss_function(outputs, labels)
        else:
            outputs, vae_out, mu, sigma = self.forward(inputs, is_validation=False)
            loss = 0.1 * self.loss_vae(vae_out, inputs, mu, sigma) + \
                self.loss_function(outputs, labels)
        self.log('train_loss', loss)
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
        self.log('val_MeanDiceScore', mean_val_dice)
        self.log('val_DiceTC', metric_tc)
        self.log('val_DiceWT', metric_wt)
        self.log('val_DiceET', metric_et)
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
                    self.model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True
                    )
        return optimizer
    
    def train_dataloader(self):
        return get_train_dataloader()
    
    def val_dataloader(self):
        return get_val_dataloader()