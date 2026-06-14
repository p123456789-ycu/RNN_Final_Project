import os
import pandas as pd

def analyze_csv_distribution(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到檔案: {csv_path}，已跳過。")
        print("-" * 50)
        return
        
    # 讀取 CSV
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    if total_samples == 0:
        print(f"📁 檔案 {csv_path} 為空白檔案。")
        print("-" * 50)
        return

    # 定義要檢查的四個診斷欄位
    diagnosis_cols = ['diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4']
    
    # 確認欄位是否存在
    for col in diagnosis_cols:
        if col not in df.columns:
            print(f"❌ 錯誤：檔案 {csv_path} 缺少 {col} 欄位。")
            return

    # 🌟 升級：嚴格多數決邏輯 (與 dataset_and_model.py 完全同步)
    def is_majority_abnormal(row):
        # 只取有醫生填寫的有效診斷
        valid_diagnoses = [val for val in row if pd.notna(val)]
        if len(valid_diagnoses) == 0:
            return False
        
        # 計算認為異常的票數
        abnormal_votes = sum([1 for val in valid_diagnoses if val != 'healthy_cough'])
        
        # 判斷：異常票數大於等於總有效票數的一半，且至少有一票
        return (abnormal_votes >= (len(valid_diagnoses) / 2.0)) and (abnormal_votes > 0)

    is_abnormal = df[diagnosis_cols].apply(is_majority_abnormal, axis=1)
    
    # 計算數量
    abnormal_count = is_abnormal.sum()
    normal_count = total_samples - abnormal_count
    
    # 計算百分比
    normal_pct = (normal_count / total_samples) * 100
    abnormal_pct = (abnormal_count / total_samples) * 100
    
    # 印出結果
    print(f"📄 資料集路徑: {csv_path}")
    print(f"   總樣本數: {total_samples} 筆")
    print(f"   🟢 正常咳嗽 (Healthy): {normal_count:4d} 筆 ({normal_pct:.2f}%)")
    print(f"   🔴 異常咳嗽 (Abnormal): {abnormal_count:4d} 筆 ({abnormal_pct:.2f}%)")
    print("-" * 50)

if __name__ == "__main__":
    target_files = [
        'train_metadata.csv',
        'val_metadata.csv',
        'test_metadata.csv'
    ]
    
    print("⏳ 開始計算資料集類別不平衡比例 (嚴格多數決版本)...\n" + "=" * 50)
    for file_name in target_files:
        analyze_csv_distribution(file_name)