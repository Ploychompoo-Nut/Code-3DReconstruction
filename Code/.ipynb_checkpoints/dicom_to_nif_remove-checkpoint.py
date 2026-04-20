import os
import dicom2nifti
import nibabel as nib
import numpy as np

# รายชื่อคนไข้ที่ต้องซ่อม
fix_list = ['0027478504', '0012203205', '0700060551', '0027744028']

base_path = '/workspace/CGH_CBCT_Data'
output_base = '/workspace/NIfTI_Output'

for patient_id in fix_list:
    print(f"\n--- Retrying Patient: {patient_id} with Ultra-Fine Extraction ---")
    
    patient_dir = os.path.join(base_path, patient_id)
    img_dicom_dir = os.path.join(patient_dir, 'no_label')
    lbl_dicom_dir = os.path.join(patient_dir, 'label')
    patient_output_dir = os.path.join(output_base, patient_id)

    img_raw_path = os.path.join(patient_output_dir, f'{patient_id}_image_raw.nii.gz')
    lbl_raw_path = os.path.join(patient_output_dir, f'{patient_id}_label_raw.nii.gz')

    try:
        # 1. แปลงไฟล์
        dicom2nifti.dicom_series_to_nifti(img_dicom_dir, img_raw_path)
        dicom2nifti.dicom_series_to_nifti(lbl_dicom_dir, lbl_raw_path)
        
        img_nii = nib.load(img_raw_path); img_data = img_nii.get_fdata()
        lbl_nii = nib.load(lbl_raw_path); lbl_data = lbl_nii.get_fdata()

        # 2. ใช้ Image Subtraction ด้วย Threshold ที่ต่ำมาก (เช่น 5 หรือ 10)
        # เราจะไม่ใช้ค่า > 1500 อีกต่อไปเพื่อเลี่ยงกระดูก
        diff = np.abs(lbl_data - img_data)
        
        # ลองไล่ระดับความละเอียด (10 -> 5 -> 2)
        thresholds = [10, 5, 2]
        mask = np.zeros(img_data.shape, dtype=np.uint8)
        
        for t in thresholds:
            temp_mask = np.where(diff > t, 1, 0).astype(np.uint8)
            pixel_count = np.sum(temp_mask)
            # ถ้าเจอพิกเซลในช่วงที่สมเหตุสมผล (3,000 - 15,000) ให้หยุดที่ค่านี้
            if 3000 < pixel_count < 20000:
                mask = temp_mask
                print(f"Success with threshold {t}!")
                break
        
        # 3. ถ้ายังหาไม่เจอจริงๆ (อาจวาดมาด้วยค่าที่เท่ากันแต่คนละสี ซึ่ง DICOM เก็บเป็นสีไม่ได้)
        # ให้เช็คว่า Max Diff คือเท่าไหร่
        if np.sum(mask) == 0:
            max_diff = np.max(diff)
            print(f"Warning: Max difference is very low ({max_diff})... trying max diff extraction.")
            if max_diff > 0:
                mask = np.where(diff >= (max_diff * 0.9), 1, 0).astype(np.uint8)

        # 4. ลบเหล็กดัดฟัน (ป้องกันไว้ก่อน)
        mask[img_data > 2800] = 0

        # 5. บันทึกผล
        final_img_path = os.path.join(patient_output_dir, f'{patient_id}_image.nii.gz')
        final_lbl_path = os.path.join(patient_output_dir, f'{patient_id}_label.nii.gz')
        
        clean_img = np.copy(img_data)
        clean_img[mask == 1] = -1000
        clean_img = np.minimum(clean_img, 1500)

        nib.save(nib.Nifti1Image(clean_img, img_nii.affine), final_img_path)
        nib.save(nib.Nifti1Image(mask, img_nii.affine), final_lbl_path)

        os.remove(img_raw_path); os.remove(lbl_raw_path)
        print(f"Final Nerve pixels found: {np.sum(mask)}")

    except Exception as e:
        print(f"Error: {e}")

print("\n--- Repair completed! ---")