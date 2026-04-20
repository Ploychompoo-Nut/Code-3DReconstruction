import os
import dicom2nifti
import nibabel as nib
import numpy as np

# 1. ระบุ Path หลัก
base_path = '/workspace/CGH_CBCT_Data'
output_base = '/workspace/NIfTI_Output'

os.makedirs(output_base, exist_ok=True)

patients = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

print(f"Found {len(patients)} patients. Starting conversion and extraction...")

for patient_id in patients:
    print(f"\n--- Processing Patient: {patient_id} ---")
    
    patient_dir = os.path.join(base_path, patient_id)
    img_dicom_dir = os.path.join(patient_dir, 'no_label')
    lbl_dicom_dir = os.path.join(patient_dir, 'label')
    
    patient_output_dir = os.path.join(output_base, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    img_path = os.path.join(patient_output_dir, f'{patient_id}_image.nii.gz')
    lbl_raw_path = os.path.join(patient_output_dir, f'{patient_id}_label_raw.nii.gz')
    final_lbl_path = os.path.join(patient_output_dir, f'{patient_id}_label.nii.gz')

    # --- ขั้นตอนที่ 1: แปลง DICOM เป็น NIfTI ---
    try:
        if os.path.exists(img_dicom_dir) and os.path.exists(lbl_dicom_dir):
            print(f"Step 1: Converting DICOM series...")
            dicom2nifti.dicom_series_to_nifti(img_dicom_dir, img_path)
            dicom2nifti.dicom_series_to_nifti(lbl_dicom_dir, lbl_raw_path)
        else:
            print(f"Skip {patient_id}: Folder no_label or label missing.")
            continue
            
        # --- ขั้นตอนที่ 2: สกัดเส้นประสาทด้วย Image Subtraction ---
        print(f"Step 2: Extracting nerve mask via subtraction...")
        
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_raw_path)
        
        img_data = img_nii.get_fdata()
        lbl_data = lbl_nii.get_fdata()
        
        # ตรวจสอบขนาดว่าตรงกันไหม (เผื่อกรณีไฟล์ DICOM ไม่ครบ)
        if img_data.shape != lbl_data.shape:
            print(f"Error: Shape mismatch for {patient_id} {img_data.shape} vs {lbl_data.shape}")
            continue

        # ลบภาพต้นฉบับออกจากภาพที่มี Label
        diff = np.abs(lbl_data - img_data)
        
        # Threshold: กรองเอาเฉพาะจุดที่ต่างกัน (เส้นประสาทที่วาดทับ)
        # ปกติเส้นที่วาดจะมีความต่างสูง (ใช้ 50-100 เป็นค่าเริ่มต้น)
        mask = np.where(diff > 50, 1, 0).astype(np.uint8)
        
        # --- ขั้นตอนที่ 3: เซฟไฟล์ Label ที่สะอาดแล้ว ---
        new_label_nii = nib.Nifti1Image(mask, img_nii.affine)
        nib.save(new_label_nii, final_lbl_path)
        
        # (Optional) ลบไฟล์ raw label ทิ้งเพื่อประหยัดพื้นที่
        if os.path.exists(lbl_raw_path):
            os.remove(lbl_raw_path)
            
        print(f"Success: Final label saved at {final_lbl_path}")
        print(f"Nerve pixels found: {np.sum(mask)}")

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")

print("\n--- All tasks completed! ---")