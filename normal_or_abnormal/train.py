import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.6):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # 計算預測機率
        probs = torch.sigmoid(logits)
        
        # pt 是模型對於「正確答案」的預測機率
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # alpha_t 根據真實標籤給予不同的基礎權重 (解決數量不平衡)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 標準的 BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 套用 Focal Loss 公式
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
# 從你剛才建好的檔案中，把我們寫好的 Dataset 和 Model 匯入進來！
from dataset_and_model import CoughDataset, CoughCareMultimodalNet, calculate_pos_weight

def train_model():
    # 1. 設定音檔資料夾路徑 (請替換成你的實際路徑)
    AUDIO_DIR = "/Users/zhanzhiya/Desktop/CoughCare_Project/coughvid_20211012"
    
    # 2. 自動計算權重 (解決類別不平衡)
    print("⏳ 計算 Loss 權重中...")
    # 🌟 修正1：直接傳入字串即可
    pos_weight_val = calculate_pos_weight('new_train_metadata.csv')

    # 3. 建立 Dataset
    print("⏳ 載入資料集中...")
    train_dataset = CoughDataset('new_train_metadata.csv', AUDIO_DIR, is_train=True)
    
    # 🌟 新增：實作 WeightedRandomSampler 解決類別不平衡
    # 先蒐集所有訓練資料的 Label
    train_labels = [train_dataset._get_label(train_dataset.df.iloc[i]) for i in range(len(train_dataset))]
    
    # 計算正常(0)與異常(1)的數量
    num_healthy = train_labels.count(0.0)
    num_abnormal = train_labels.count(1.0)
    
    # 計算每種類別的權重 (數量越少，權重越高)
    weight_healthy = 1.0 / num_healthy
    weight_abnormal = 1.0 / num_abnormal
    
    # 為訓練集中的「每一筆資料」分配權重
    samples_weight = torch.tensor([weight_abnormal if label == 1.0 else weight_healthy for label in train_labels])
    
    # 建立 Sampler
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    # 🌟 建立 DataLoader (注意：使用了 sampler 就不能設定 shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    
    # 驗證集不需要 Sampler，保持原樣
    val_dataset = CoughDataset('new_val_metadata.csv', AUDIO_DIR, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 3. 初始化模型、運算設備
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ 使用運算設備: {device}")
    
    print("⏳ 初始化多模態融合大腦 (ResNet + MLP)...")
    model = CoughCareMultimodalNet().to(device)

    # 🌟 補回遺失的 Loss Function 設定
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # 權重衰減策略不變
    optimizer = torch.optim.Adam([
        {'params': [p for name, p in model.named_parameters() if 'resnet' in name], 'lr': 1e-5},
        {'params': model.vision_proj.parameters(), 'lr': 1e-3},
        {'params': model.tabular_mlp.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-2)

    # 🌟 冠軍設定 3：學習率排程器 (沒有 verbose 參數，避免 PyTorch 報錯)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 5. 開始訓練迴圈
    epochs = 15
    print("🚀 開始訓練...")
    best_val_auc = 0.0  # 🌟 改成記錄最佳 AUC (越高越好，所以初始為 0)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (mels, tabs, labels) in enumerate(train_loader):
            mels, tabs, labels = mels.to(device), tabs.to(device), labels.to(device).unsqueeze(1) 
            
            optimizer.zero_grad()
            outputs = model(mels, tabs) # 同時餵入圖片與病歷
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        # 計算這個 Epoch 的平均 Loss
        epoch_avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} 完成! 平均 Loss: {epoch_avg_loss:.4f}")
        print("-" * 30)

        # 在每個 epoch 的訓練迴圈結束後加入：
        # ==========================================
        # 🌟 全新升級的 Validation 迴圈 (收集分數算 AUC)
        # ==========================================
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_scores = []
        
        with torch.no_grad():
            for mels, tabs, labels in val_loader:
                mels, tabs, labels = mels.to(device), tabs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(mels, tabs) # 同時餵入圖片與病歷
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 收集分數與標籤用來算 AUC
                scores = torch.sigmoid(outputs).squeeze().cpu().tolist()
                
                # 處理 batch size = 1 的邊界情況
                if isinstance(scores, float):
                    scores = [scores]
                    
                all_val_scores.extend(scores)
                # 轉回一維列表儲存
                all_val_labels.extend(labels.squeeze(1).cpu().tolist())

        epoch_val_loss = val_loss / len(val_loader)
        # 🌟 計算這一代 Validation 的 AUC
        epoch_val_auc = roc_auc_score(all_val_labels, all_val_scores)
        
        print(f"🔍 Epoch {epoch+1} 驗證 Loss: {epoch_val_loss:.4f} | 驗證 AUC: {epoch_val_auc:.4f}")

        # 1. 讓排程器看 Validation Loss 
        # (💡 提示：Loss 的變化比較平滑，用它來決定何時降學習率比 AUC 更穩定)
        scheduler.step(epoch_val_loss)

        # 2. 🌟 改用 AUC 當作儲存最佳模型的標準！(大於 best_val_auc 才存檔)
        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            torch.save(model.state_dict(), 'best_coughcare_model.pth')
            print(f"🌟 發現新高分 (AUC: {epoch_val_auc:.4f})！最佳模型已更新。")
        print("-" * 30)

    # 訓練結束，存檔
    torch.save(model.state_dict(), 'coughcare_model.pth')
    print("💾 模型已成功儲存為 coughcare_model.pth！")

if __name__ == "__main__":
    train_model()