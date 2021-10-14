# SegTransVAE: Hybrid CNN - Transformer with Regularization for medical image segmentation
This repo is the official implementation for SegTransVAE. 

## SegTransVAE 
Here is the network architecture of SegTransVAE, a hybrid CNN-Transformer-VAE for medical image segmentation  
![SegTransVAE](figure/SegTransVAE.png)

## Requirements
- python 3.8
- pytorch 1.8.0
- monai 0.6
- pytorch lightning
- nibabel
- itk 


## Data 
After downloading BraTS 2021 dataset, create a json file, which contains the path to images and the correspoding labels as follow
<!-- ![Data configuration](figure/data.png) -->
<p align="center">
<img src="figure/data.png" width=70% height=70%>
</p>
Then, use data/brats.py file to read the data by monai. 

## Training
Run the training script on BraTS dataset. Change the model hyperparameters setup in trainer.py. When everything is finish, go to **lightning_train.py** to config multi-gpu training or half-precision training. 

``` 
python lightning_train.py --exp EXP 
```
EXP is the name of the experiment and it will save all the checkpoint and the logs based on that experiment name. 

## Testing 
After finish training, you can test your model by replace the path to the checkpoints in **lightning_test.py**
```
python lightning_test.py
```
Then the evaluation metric will print in the terminal, including dice score and 95% hausdorff distance. 


## Quantitative result
Quantitive comparison of performance on BraTS 2021 (our test set)
<!-- ![table_brats](figure/table_brats.png) -->

<p align="center">
<img src="figure/table_brats.png" width=70% height=70%>
</p>

Quantitive comparison of performance on KiTS19 with 5-fold cross validation. 
<!-- ![table_kidney](figure/table_kidney.png) -->
<p align="center">
<img src="figure/table_kidney.png" width=70% height=70%>
</p>
<!-- ![table_tumor](figure/table_kidneytumor.png) -->
<p align="center">
<img src="figure/table_kidneytumor.png" width=70% height=70%>
</p>

## Visual Comparision with SOTA Methods 
Visual Comparision of our method on BraTS 2021 and KiTS19 dataset with 3D U-Net, SegresnetVAE and UNETR. 
![brats](figure/brats.png)

![kits](figure/kits.png)

## Complexity 
The complexity of SegTransVAE is compared to other models in terms of the number of parameters and the averaged inference time. The benchmark is calculated based on the input size of (4, 128, 128, 128)
<!-- ![complexity](figure/complexity.png) -->

<p align="center">
<img src="figure/complexity.png" width=70% height=70%>
</p>
