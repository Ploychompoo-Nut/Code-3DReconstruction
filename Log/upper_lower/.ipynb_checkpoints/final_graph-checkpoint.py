import pandas as pd
import matplotlib.pyplot as plt

# 1. อ่านไฟล์ CSV ล่าสุด
df = pd.read_csv('training_log.csv')

plt.figure(figsize=(10, 6))

# วาดเส้น Loss (สีน้ำเงิน)
plt.plot(df['epoch'], df['loss'], label='Train Loss', color='#1f77b4', linewidth=2)

# วาดเส้น Accuracy (สีส้ม)
plt.plot(df['epoch'], df['acc'], label='Train acc', color='#ff7f0e', linewidth=2)

plt.title('Training accuracy & loss (Combined)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# เซฟรูปออกมาเป็นไฟล์คุณภาพสูง
plt.savefig('accuracy_loss_final_report.png', dpi=120)
plt.show()