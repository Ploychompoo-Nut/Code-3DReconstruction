import os
import nibabel as nib

# กำหนด Path หลัก
data_path = "/workspace/PMCSeg/dataset"
patients = [f for f in os.listdir(data_path) if f.startswith('Patient')]

# รายชื่อโฟลเดอร์ที่ต้องการตรวจสอบภายในแต่ละ Patient
# sub_folders = ["Upper", "Lower", "Skull"]
sub_folders = ["Upper"]

print(f"Total patients found: {len(patients)}")
print("-" * 30)

for p in patients:
    print(f"Checking {p}...")
    for folder in sub_folders:
        # สร้าง Path สำหรับ image.nii.gz ในแต่ละโฟลเดอร์ย่อย
        img_file = os.path.join(data_path, p, folder, "image.nii.gz")
        
        if os.path.exists(img_file):
            try:
                vol = nib.load(img_file)
                print(f"  [OK] {folder}: {vol.shape}")
            except Exception as e:
                print(f"  [ERROR] {folder}: Cannot read file! ({e})")
        else:
            print(f"  [MISSING] {folder}: image.nii.gz not found!")
    print("-" * 30)