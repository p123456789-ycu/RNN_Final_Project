import pandas as pd
from sklearn.model_selection import train_test_split

# ⚠️ 請將這裡替換成你最原始 2 萬多筆的 CSV 檔名
RAW_CSV_PATH = "/Users/zhanzhiya/Desktop/CoughCare_Project/coughvid_20211012/metadata_compiled.csv" 

print("⏳ 開始讀取 2 萬多筆的原始資料...")
df = pd.read_csv(RAW_CSV_PATH)
print(f"   原始資料總數: {len(df)} 筆")

# ==========================================
# 1. 第一層篩選：只要有「任何一位」專家標註過，就保留
# ==========================================
diagnosis_cols = ['diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4']
has_any_diagnosis = df[diagnosis_cols].notna().any(axis=1)
filtered_df = df[has_any_diagnosis]
print(f"   第一階段清洗：保留具備有效標註的資料共 {len(filtered_df)} 筆")

# ==========================================
# 2. 第二層篩選：抽出黃金測試集 (四位專家都有標註)
# ==========================================
golden_mask = filtered_df[diagnosis_cols].notna().all(axis=1)
golden_df = filtered_df[golden_mask]

# ==========================================
# 3. 處理剩餘資料 (1~3 位專家標註)：切分訓練與驗證
# ==========================================
rest_df = filtered_df[~golden_mask]
train_df, val_df = train_test_split(rest_df, test_size=0.2, random_state=42)

# ==========================================
# 4. 輸出存檔
# ==========================================
golden_df.to_csv('golden_test_metadata.csv', index=False)
train_df.to_csv('new_train_metadata.csv', index=False)
val_df.to_csv('new_val_metadata.csv', index=False)

print("\n========== 終極切分結果 ==========")
print(f"🌟 黃金測試集 (期末考卷): {len(golden_df)} 筆 -> golden_test_metadata.csv")
print(f"📚 訓練集 (日常練習):   {len(train_df)} 筆 -> new_train_metadata.csv")
print(f"🔍 驗證集 (模擬考):     {len(val_df)} 筆 -> new_val_metadata.csv")
print("✅ 資料處理完畢，請修改 train.py 的路徑後開始訓練！")