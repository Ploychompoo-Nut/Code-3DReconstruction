# -*- coding: utf-8 -*-

import torch
import monai.losses

from torch import nn
from torch.nn.functional import max_pool3d
from monai.losses import DiceLoss

class crossentry(nn.Module):#实现交叉熵损失

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):#模型的前向传播
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth))#y_pred 应该是模型输出的概率值，但在交叉熵损失中，我们通常对 y_pred 使用 softmax 或 sigmoid 来确保其在 [0, 1] 之间。


class cross_loss(nn.Module):#现标准的二分类交叉熵损失。

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + smooth))#它同时考虑了 y_true 为正和为负的情况。使用了 smooth 来确保数值稳定性。


class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, label, output, class_index=1):
        eps = 1e-6
        output = torch.softmax(output, dim=1)
        # 选择感兴趣的类别通道
        label = label[:, 1, ...]
        # print(label)
        output = output[:, 1, ...]
        # print(output)

        #将输入和标签展平为一维张量，以便于计算
        output = output.contiguous().view(-1)
        label = label.contiguous().view(-1)
        # 计算交集
        intersection = (output * label).sum()

        # 计算label和output中值为1的元素的和
        label_sum = label.sum()
        output_sum = output.sum()

        # 计算Dice系数
        dice = (2. * intersection + eps) / (label_sum + output_sum + eps)

        ## Compute Dice loss
        return 1-dice


'''
Another Loss Function proposed by us in IEEE transactions on Image Precessing:
Paper: https://ieeexplore.ieee.org/abstract/document/9611074
Code: https://github.com/YaoleiQi/Examinee-Examiner-Network
'''


class Dropoutput_Layer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = (
            -((torch.sum(w * y_true * torch.log(y_pred + smooth)) /
               torch.sum(w * y_true + smooth)) +
              (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) /
               torch.sum(w * (1 - y_true) + smooth))) / 2)
        return loss_ce


import torch
import torch.nn.functional as F

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self,weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.weight is not None:
            self.weight = self.weight.to(device)

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight)

# class CombinedLoss(torch.nn.Module):
#     def __init__(self,weight):
#         super(CombinedLoss, self).__init__()
#         self.weight = weight
#         self.weighted_ce = WeightedCrossEntropyLoss(weight)
#         self.dice_loss = dice_loss()
#
#     def forward(self, inputs, targets,n):
#         n=float(n*0.002)
#         print(f"n:,{n}")
#         ce_loss = self.weighted_ce(inputs, targets)
#         dice_loss = self.dice_loss(inputs, targets)
#         return n*ce_loss + (1-n)*dice_loss


# ในไฟล์ losses/Loss.py
class CombinedLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(CombinedLoss, self).__init__()
        # ใช้ DiceLoss จาก monai โดยตรงเพื่อให้ค่าแม่นยำและไม่ติดลบ
        self.dice_loss = DiceLoss(
            include_background=False, # สำคัญมาก: โฟกัสแค่เส้นเลือด (Class 1)
            softmax=True,            # ให้ MONAI จัดการ Softmax ข้างในเองเลย
            to_onehot_y=False        # เพราะ Dataloader ทำ onehot มาแล้ว
        )
        self.weighted_ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets, gt_dist=None):
        # inputs: [B, 2, 96, 96, 96], targets: [B, 2, 96, 96, 96]
        
        # 1. Dice Loss
        d_loss = self.dice_loss(inputs, targets)
        
        # 2. Cross Entropy Loss (ต้องการ target เป็น long index)
        ce_target = torch.argmax(targets, dim=1) # [B, 96, 96, 96]
        ce_loss = self.weighted_ce(inputs, ce_target)
        
        # รวมกันเพื่อให้ได้ค่าที่สมดุล
        return ce_loss + d_loss
