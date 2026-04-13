import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandRotated,
    RandFlipd,
    RandGaussianNoised,
    ToTensord,
    EnsureChannelFirstd
)

transform_img_lab = Compose([
    # ลบ EnsureChannelFirstd ออก เพราะใน Dataloader คุณทำ np.newaxis ไปแล้ว
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotated(
        keys=["image", "label"], 
        range_x=0.26, range_y=0.26, range_z=0.26, 
        prob=0.2, mode=("bilinear", "nearest")
    ),
    ToTensord(keys=["image", "label"]),
])