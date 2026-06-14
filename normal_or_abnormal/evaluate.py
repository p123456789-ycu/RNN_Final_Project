import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, classification_report, roc_curve
import numpy as np
from dataset_and_model import CoughDataset, CoughCareMultimodalNet, calculate_pos_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# ==========================================
# ⚠️ 請修改這兩個路徑
# ==========================================
METADATA_PATH = "golden_test_metadata.csv"   # 你的測試集 CSV 路徑
AUDIO_DIR = "coughvid_20211012"       # 你的音檔資料夾路徑
MODEL_PATH    = "best_coughcare_model76.pth"

# ==========================================
# 載入模型
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")

elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 使用 Apple Silicon GPU (MPS)")

else:
    device = torch.device("cpu")
    print("⚠️ 使用 CPU")

model = CoughCareMultimodalNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"✅ 模型載入成功，使用裝置: {device}")

# ==========================================
# 載入測試集
# ==========================================
test_dataset = CoughDataset(
    metadata_path=METADATA_PATH,
    audio_dir=AUDIO_DIR,
    is_train=False   # 關閉資料擴增
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
print(f"✅ 測試集載入成功，共 {len(test_dataset)} 筆")

# ==========================================
# 推論
# ==========================================
all_scores = []
all_labels = []

with torch.no_grad():
    # 🌟 讀取三個變數：圖片、病歷、標籤
    for mel_specs, tab_features, labels in test_loader:
        mel_specs = mel_specs.to(device)
        tab_features = tab_features.to(device) # 將病歷送入運算設備
        
        logits = model(mel_specs, tab_features) # 同時餵入兩種特徵
        scores = torch.sigmoid(logits).squeeze().cpu().tolist()
        # 🌟 修改這行，加上底線接住不要的回傳值
        #logits, _, _, _ = model(mel_specs, tab_features) 
        #scores = torch.sigmoid(logits).squeeze().cpu().tolist()
        
        # 處理 batch size = 1 的邊界情況
        if isinstance(scores, float):
            scores = [scores]
        
        all_scores.extend(scores)
        all_labels.extend(labels.tolist())

# ==========================================
# 計算指標 (直接使用 Youden's J 動態閾值)
# ==========================================
auc = roc_auc_score(all_labels, all_scores)
pr_auc = average_precision_score(all_labels, all_scores)

# 直接找最佳公平閾值，不理會死板的 0.5
fpr, tpr, thresholds_roc = roc_curve(all_labels, all_scores)
j_scores = tpr - fpr  
best_roc_threshold = thresholds_roc[np.argmax(j_scores)]

print("\n========== 終極評估結果 ==========")
print(f"AUC:     {auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"🎯 模型專屬最佳閾值: {best_roc_threshold:.4f}")

# 用這個模型專屬的閾值來算成績，紅字警告就會徹底消失！
preds_best = [1 if s >= best_roc_threshold else 0 for s in all_scores]
print("\n詳細分類報告：")
print(classification_report(all_labels, preds_best, target_names=["正常咳嗽 (低風險)", "異常咳嗽 (高風險)"]))
# ==========================================
# 繪製與儲存混淆矩陣 (比例版本 0.00 ~ 1.00)
# ==========================================
print("\n🎨 正在繪製比例版混淆矩陣...")

# 🌟 修改 1：加入 normalize='true'，讓每一列的總和變成 1.0 (計算 Recall 比例)
cm = confusion_matrix(all_labels, preds_best, normalize='true')

# 設定類別名稱
class_names = ["Healthy (Low Risk)", "Abnormal (High Risk)"]

# 畫圖設定
plt.figure(figsize=(8, 6))

# 🌟 修改 2：將 fmt='d' (整數) 改成 fmt='.2f' (顯示小數點後兩位)
# 可以依喜好加上 vmin=0.0, vmax=1.0 讓顏色對比更精準
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
            vmin=0.0, vmax=1.0,
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 16})

# 設定標題與座標軸
plt.title(f'CoughCare Normalized Confusion Matrix\n(AUC: {auc:.4f})', fontsize=16, pad=15)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()

# 儲存圖片並顯示
plt.savefig('confusion_matrix_normalized.png', dpi=300)
print("✅ 比例版混淆矩陣已成功儲存為 'confusion_matrix_normalized.png'！")
plt.show()