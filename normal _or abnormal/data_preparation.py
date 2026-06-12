import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_coughvid_data(metadata_path):
    print("Step 1: 讀取 CSV 檔案...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"找不到檔案 {metadata_path}，請確認路徑是否正確。")
        
    # 讀取全部資料
    df = pd.read_csv(metadata_path)
    print(f"總資料筆數: {len(df)}")

    # Step 2: 依照 uuid 進行 Subject-level 切分 (直接使用完整資料集)
    # 2.1 先找出所有不重複的 uuid
    unique_uuids = df['uuid'].unique()
    print(f"獨立使用者 (uuid) 總數: {len(unique_uuids)}")

    # 2.2 切分 uuid: 先切出 Train (70%) 和 Temp (30%)
    train_uuids, temp_uuids = train_test_split(unique_uuids, test_size=0.30, random_state=42)
    
    # 2.3 再將 Temp (30%) 對半切分成 Val (15%) 和 Test (15%)
    val_uuids, test_uuids = train_test_split(temp_uuids, test_size=0.50, random_state=42)

    # 2.4 根據切分好的 uuid 集合，把原始 DataFrame 拆開
    train_df = df[df['uuid'].isin(train_uuids)]
    val_df = df[df['uuid'].isin(val_uuids)]
    test_df = df[df['uuid'].isin(test_uuids)]

    # === 安全檢查 (Sanity Check) ===
    # 確保 Train 和 Test 的 uuid 絕對沒有交集
    train_set = set(train_uuids)
    val_set = set(val_uuids)
    test_set = set(test_uuids)
    
    assert len(train_set.intersection(test_set)) == 0, "⚠️ 警告：Train 和 Test 發生 Data Leakage！"
    assert len(train_set.intersection(val_set)) == 0, "⚠️ 警告：Train 和 Val 發生 Data Leakage！"
    print("✅ 資料洩漏檢查通過：各資料集的 uuid 無重疊。")

    # 印出最終切分結果
    print("-" * 30)
    print(f"Train Set: {len(train_df)} 筆音檔 (約 {len(train_df)/len(df):.1%})")
    print(f"Val Set  : {len(val_df)} 筆音檔 (約 {len(val_df)/len(df):.1%})")
    print(f"Test Set : {len(test_df)} 筆音檔 (約 {len(test_df)/len(df):.1%})")
    print("-" * 30)

    # Step 3: 儲存成新的 CSV 檔案
    train_df.to_csv('train_metadata.csv', index=False)
    val_df.to_csv('val_metadata.csv', index=False)
    test_df.to_csv('test_metadata.csv', index=False)
    print("💾 已成功儲存 train_metadata.csv, val_metadata.csv, test_metadata.csv！")

# 執行腳本
if __name__ == "__main__":
    prepare_coughvid_data("/Users/zhanzhiya/Desktop/CoughCare_Project/all_4_diagnosis.csv")