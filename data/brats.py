from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandSpatialCropd,
    RandFlipd,
    MapTransform,
    NormalizeIntensityd, 
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    CenterSpatialCropd,
)
from monai.data import DataLoader, Dataset
import numpy as np
import json
set_determinism(seed=0)
with open('/claraDevDay/Data/Brats2021/brats2021_fullpath.json') as f:
    data = json.load(f)
train_files, val_files, test_files = data['training'], data['validation'], data['testing']
print(len(train_files), len(val_files), len(test_files))

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
 
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 4 to construct TC
            result.append(np.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 4 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 1, d[key] == 4), d[key] == 2
                )
            )
            # label 4 is ET
            result.append(d[key] == 4)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
def get_train_dataloader():

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys = ['label']),
            RandSpatialCropd(keys=["image", "label"],
                            roi_size = [128,128,128], 
                            #  roi_size = [96,96,96],
                            random_size = False),
            RandFlipd(keys = ["image", "label"],
                     prob = 0.5,
                     spatial_axis = 0),
            RandFlipd(keys = ["image", "label"],
                     prob = 0.5,
                     spatial_axis = 1),
            RandFlipd(keys = ["image", "label"],
                     prob = 0.5,
                     spatial_axis = 2),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            RandScaleIntensityd(keys = "image", prob = 1, factors = 0.1),
            RandShiftIntensityd(keys = "image", prob = 1, offsets = 0.1),
            ToTensord(keys=["image", "label"]),
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

    
    return train_loader

def get_val_dataloader():
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(
                keys = ['label']),
            CenterSpatialCropd(keys=["image", "label"],
                            roi_size = [128,128,128], 
                            ),
            NormalizeIntensityd(keys = "image",
                               nonzero = True,
                               channel_wise = True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    return val_loader