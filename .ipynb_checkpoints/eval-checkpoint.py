import torch
import os
import nibabel as nib
import numpy as np
from statistics import mean, stdev
from medpy.metric.binary import hd95, asd
from skimage import measure

class Eval:
    def __init__(self, classes=1):
        self.iou_list, self.dice_list = [], []
        self.hd95_list, self.asd_list, self.rve_list = [], [], []

    def reset_eval(self):
        self.iou_list.clear(); self.dice_list.clear()
        self.hd95_list.clear(); self.asd_list.clear(); self.rve_list.clear()

    def get_lcc(self, image):
        """ เลือกเฉพาะกลุ่มก้อนที่ใหญ่ที่สุด (ช่วยลดค่า HD95/ASD จาก Noise) """
        labels = measure.label(image)
        if labels.max() == 0:
            return image
        main_blob = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return main_blob.astype(np.uint8)

    def compute_metrics(self, pred, gt, voxel_spacing=(1, 1, 1), print_val=False):
        # 1. เตรียมข้อมูล (ใช้ float เพื่อป้องกันค่า overflow ตอนลบกัน)
        pred_np = pred.detach().cpu().numpy().squeeze()
        gt_np = gt.detach().cpu().numpy().squeeze()
        
        pred_bin = (pred_np > 0.5).astype(np.uint8)
        gt_bin = (gt_np > 0.5).astype(np.uint8)
        
        # กรอง Noise ด้วย Largest Connected Component
        pred_bin = self.get_lcc(pred_bin)

        # คำนวณปริมาตรแบบ float เพื่อความแม่นยำ
        vol_pred = float(pred_bin.sum())
        vol_gt = float(gt_bin.sum())

        # 2. คำนวณ DSC & IoU
        intersection = float(np.logical_and(gt_bin, pred_bin).sum())
        union = float(np.logical_or(gt_bin, pred_bin).sum())
        
        dice = (2. * intersection) / (vol_gt + vol_pred + 1e-6)
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # 3. คำนวณ RVE (Relative Volume Error)
        rve = abs(vol_pred - vol_gt) / (vol_gt + 1e-6)

        # 4. คำนวณ HD95 และ ASD (Physical distance ในหน่วย mm)
        h95_val, asd_val = 0.0, 0.0
        if vol_pred > 0 and vol_gt > 0:
            h95_val = hd95(pred_bin, gt_bin, voxelspacing=voxel_spacing)
            asd_val = asd(pred_bin, gt_bin, voxelspacing=voxel_spacing)
            self.hd95_list.append(h95_val)
            self.asd_list.append(asd_val)

        self.dice_list.append(dice)
        self.iou_list.append(iou)
        self.rve_list.append(rve)

        if print_val:
            # โชว์ครบทั้ง 5 ค่าตามที่นัทต้องการ
            print(f'DSC: {dice:.4f} | IoU: {iou:.4f} | HD95: {h95_val:.4f} | ASD: {asd_val:.4f} | RVE: {rve:.4f}')

    def mean_metric(self):
        def stats(l): 
            if len(l) == 0: return 0.0, 0.0
            m = mean(l)
            s = stdev(l) if len(l) > 1 else 0.0
            return m, s
            
        return {
            "DSC": stats(self.dice_list),
            "IoU": stats(self.iou_list),
            "HD95": stats(self.hd95_list),
            "ASD": stats(self.asd_list),
            "RVE": stats(self.rve_list)
        }

if __name__ == "__main__":
    # --- ปรับ Path ตรงนี้ให้ตรงตามแต่ละโฟลเดอร์โมเดล ---
    GT_DIR = "./dataset/test/label/" 
    PRED_DIR = "./Results/upper_lower/UNETR_max/" 

    evaluator = Eval()
    files = sorted([f for f in os.listdir(GT_DIR) if f.endswith('.nii.gz')])
    print(f"Found {len(files)} files. Starting evaluation...")

    for f in files:
        pred_p = os.path.join(PRED_DIR, f)
        if os.path.exists(pred_p):
            gt_nii = nib.load(os.path.join(GT_DIR, f))
            pred_nii = nib.load(pred_p)
            
            # ดึง Spacing (mm) มาใช้คำนวณ HD95/ASD ให้แม่นยำ
            spacing = gt_nii.header.get_zooms() 
            
            evaluator.compute_metrics(
                torch.from_numpy(pred_nii.get_fdata()), 
                torch.from_numpy(gt_nii.get_fdata()), 
                voxel_spacing=spacing, 
                print_val=True
            )
        else:
            print(f"File not found: {f}")

    # สรุปผลลัพธ์ลงตาราง
    res = evaluator.mean_metric()
    
    print("\n" + "="*100)
    print(f"{'Method':<12} | {'DSC':<15} | {'IoU':<15} | {'HD95(mm)':<15} | {'ASD(mm)':<15} | {'RVE':<15}")
    print("-" * 100)
    print(f"{'SegResNet':<12} | "
          f"{res['DSC'][0]:.4f}±{res['DSC'][1]:.4f} | "
          f"{res['IoU'][0]:.4f}±{res['IoU'][1]:.4f} | "
          f"{res['HD95'][0]:.4f}±{res['HD95'][1]:.4f} | "
          f"{res['ASD'][0]:.4f}±{res['ASD'][1]:.4f} | "
          f"{res['RVE'][0]:.4f}±{res['RVE'][1]:.4f}")
    print("="*100)