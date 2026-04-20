import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. ระบุ Path (อ้างอิงจาก NIfTI_Output ที่เราทำเสร็จแล้ว)
output_base = '/workspace/NIfTI_Output'
patients = sorted([f for f in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, f))])

# 2. ตั้งค่าการแสดงผล (แถวละ 5 คน มี 2 แถว)
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(25, 20))
axes = axes.flatten()

print(f"Generating summary for {len(patients)} patients...")

for i, patient_id in enumerate(patients):
    img_path = os.path.join(output_base, patient_id, f'{patient_id}_image.nii.gz')
    lbl_path = os.path.join(output_base, patient_id, f'{patient_id}_label.nii.gz')
    
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        # โหลดข้อมูล
        img_data = nib.load(img_path).get_fdata()
        lbl_data = nib.load(lbl_path).get_fdata()
        
        # หา Slice ที่มี Label มากที่สุด (จะได้เห็นเส้นประสาทชัดๆ)
        label_counts = np.sum(lbl_data, axis=(0, 1))
        best_slice = np.argmax(label_counts)
        
        # แสดงภาพ CT (ที่ถูก Clean แล้ว) พร้อม Overlay Label
        ax = axes[i]
        ax.imshow(img_data[:, :, best_slice].T, cmap='gray', origin='lower')
        # แสดง Mask สีแดงทับลงไป
        ax.imshow(lbl_data[:, :, best_slice].T, cmap='Reds', alpha=0.5, origin='lower')
        
        ax.set_title(f"ID: {patient_id}\nSlice: {best_slice}\nPixels: {int(np.sum(lbl_data))}")
        ax.axis('off')
    else:
        axes[i].text(0.5, 0.5, f"Missing: {patient_id}", ha='center')
        axes[i].axis('off')

# ซ่อนช่องที่เหลือที่ไม่มีข้อมูล
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('/workspace/Dataset_Summary.png') # บันทึกเป็นรูปไฟล์เดียวดูง่ายๆ
plt.show()

print("Summary saved as /workspace/Dataset_Summary.png")