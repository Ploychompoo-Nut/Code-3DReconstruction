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

def transform_img_lab(args):
    """
    สร้าง Pipeline สำหรับการทำ Data Augmentation พื้นฐาน
    เพื่อให้โมเดลทำงานต่อได้
    """
    return Compose([
        # ตรวจสอบลำดับ Channel (เพื่อให้แน่ใจว่าเป็น [C, H, W, D])
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # การสุ่ม Flip (พลิกภาพ)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        
        # การสุ่มหมุนภาพเล็กน้อย (ประมาณ 15 องศา)
        RandRotated(
            keys=["image", "label"], 
            range_x=0.26, range_y=0.26, range_z=0.26, 
            prob=0.2, mode=("bilinear", "nearest")
        ),
        
        # แปลงเป็น Tensor เพื่อเข้า GPU
        ToTensord(keys=["image", "label"]),
    ])