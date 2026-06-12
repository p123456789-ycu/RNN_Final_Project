import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import torch.nn as nn
import random
import librosa
from torchvision import models

class CoughDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, audio_dir, is_train=False, target_sec=2.0, sr=16000):
        self.df = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.is_train = is_train
        self.target_sr = sr
        self.target_length = int(target_sec * sr) 

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=320, n_mels=128
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    # 🌟 嚴格多數決標籤
    def _get_label(self, row):
        diagnoses = [row['diagnosis_1'], row['diagnosis_2'], row['diagnosis_3'], row['diagnosis_4']]
        valid_diagnoses = [d for d in diagnoses if pd.notna(d)]
        if len(valid_diagnoses) == 0:
            return 0.0
        abnormal_votes = sum([1 for d in valid_diagnoses if d != 'healthy_cough'])
        if abnormal_votes >= (len(valid_diagnoses) / 2.0) and abnormal_votes > 0:
            return 1.0
        return 0.0

    # 🌟 新增：提取病患結構化數據 (Tabular Features)
    def _get_tabular_features(self, row):
        # 1. 年齡 (除以 100 正規化，若缺漏補 40 歲)
        age = float(row['age']) if pd.notna(row['age']) else 40.0
        age_norm = age / 100.0
        
        # 2. 性別 (男性=1, 女性=0, 未知=0.5)
        gender = str(row['gender']).lower()
        if gender == 'male': g_val = 1.0
        elif gender == 'female': g_val = 0.0
        else: g_val = 0.5
        
        # 3. 呼吸道病史 (True=1, 否則=0)
        resp_cond = str(row['respiratory_condition']).lower()
        resp = 1.0 if resp_cond == 'true' else 0.0
        
        # 4. 發燒/肌肉痠痛 (True=1, 否則=0)
        fever_cond = str(row['fever_muscle_pain']).lower()
        fever = 1.0 if fever_cond == 'true' else 0.0
        
        return torch.tensor([age_norm, g_val, resp, fever], dtype=torch.float32)

    # 🌟 結合 Librosa 的精準裁切
    def _crop_with_librosa(self, waveform):
        y = waveform.squeeze(0).numpy()
        intervals = librosa.effects.split(y, top_db=25)
        current_len = waveform.shape[1]
        
        if len(intervals) == 0:
            start_idx = max(0, (current_len - self.target_length) // 2)
            end_idx = start_idx + self.target_length
        else:
            best_interval = max(intervals, key=lambda i: np.sum(y[i[0]:i[1]]**2))
            center_idx = (best_interval[0] + best_interval[1]) // 2
            start_idx = center_idx - (self.target_length // 2)
            end_idx = start_idx + self.target_length
            
        if start_idx < 0:
            start_idx = 0
            end_idx = self.target_length
        elif end_idx > current_len:
            end_idx = current_len
            start_idx = current_len - self.target_length
            
        cropped = waveform[:, max(0, start_idx):min(current_len, end_idx)]
        if cropped.shape[1] < self.target_length:
            pad_amount = self.target_length - cropped.shape[1]
            cropped = torch.nn.functional.pad(cropped, (0, pad_amount))
        return cropped

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['uuid']
        audio_path = os.path.join(self.audio_dir, f"{file_name}.webm") 
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, f"{file_name}.ogg")
            
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self.target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)(waveform)

        # 聲音裁切
        waveform = self._crop_with_librosa(waveform)

        if self.is_train:
            if random.random() < 0.5: waveform = waveform + torch.randn_like(waveform) * 0.005 
            if random.random() < 0.5:
                shift_amt = int(random.uniform(-0.1, 0.1) * self.target_sr)
                waveform = torch.roll(waveform, shifts=shift_amt, dims=1)
            if random.random() < 0.3:
                mask_len = int(random.uniform(0.05, 0.15) * self.target_sr)
                mask_start = random.randint(0, waveform.shape[1] - mask_len)
                waveform[:, mask_start:mask_start+mask_len] = 0.0

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)

        # 🌟 一次打包回傳：圖片、病歷、標籤
        tab_features = self._get_tabular_features(row)
        label = torch.tensor(self._get_label(row), dtype=torch.float32)
        
        return mel_spec, tab_features, label

# ==========================================
# 自動計算權重 (嚴格多數決版本)
# ==========================================
def calculate_pos_weight(metadata_path):
    df = pd.read_csv(metadata_path)
    
    def is_majority_abnormal(row):
        diagnoses = [row['diagnosis_1'], row['diagnosis_2'], row['diagnosis_3'], row['diagnosis_4']]
        valid = [d for d in diagnoses if pd.notna(d)]
        if len(valid) == 0: 
            return False
        abnormal_votes = sum([1 for d in valid if d != 'healthy_cough'])
        return (abnormal_votes >= (len(valid) / 2.0)) and (abnormal_votes > 0)
        
    is_abnormal = df.apply(is_majority_abnormal, axis=1)
    num_abnormal = is_abnormal.sum()
    num_healthy = len(df) - num_abnormal
    
    # 正常數量 / 異常數量
    return num_healthy / (num_abnormal + 1e-5)

# ==========================================
# 終極武器：多模態融合網路 (Multimodal Fusion Net)
# ==========================================
class CoughCareMultimodalNet(nn.Module):
    def __init__(self, num_tabular_features=4):
        super(CoughCareMultimodalNet, self).__init__()
        
        # 1. 視覺大腦 (看頻譜圖)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # 拔掉最後的分類頭，只取特徵
        
        self.vision_proj = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 2. 病歷大腦 (看結構化數據)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 3. 決策大腦 (將視覺與病歷特徵結合)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64), # 128(視覺) + 16(病歷)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        # 處理頻譜圖
        img = img.expand(-1, 3, -1, -1)
        vis_feat = self.resnet(img)
        vis_feat = self.vision_proj(vis_feat)
        
        # 處理病歷
        tab_feat = self.tabular_mlp(tab)
        
        # 拼接 (Fusion)
        combined = torch.cat((vis_feat, tab_feat), dim=1)
        
        return self.classifier(combined)