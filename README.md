# CoughCare, Rag, Ollama
結合 RAG 技術與 Ollama 本地大型語言模型的智慧咳嗽分析Agent
---
## normal _or abnormal

### 正常 vs 異常

| Label | Define |
| --- | --- |
| normal | healthy_cough |
| abnormal | upper_infection, lower_infection, obstructive_disease, COVID-19 |

---

## CoughNet

### Covid vs Healthy vs Symptomatic

| Label | Define |
| --- | --- |
| covid | COVID-19 |
| healthy | healthy_cough |
| symptomatic | upper_infection, lower_infection, obstructive_disease |

#### Data

| Data | Content |
| --- | --- |
| Train Data | `diagnosis_1` ～ `diagnosis_4` |
| Test Data | `status_SSL` |

---

## Two stage

臨床特徵：`age`、`gender`

特徵向量（516 維）：
- `emb_0` ~ `emb_511`：HeAR embedding（512維）
- `age`：年齡（數值，標準化）
- `gender`：性別（male=0, female=1, other=2）
- `respiratory_condition`：慢性呼吸道疾病（0/1）
- `fever_muscle_pain`：發燒或肌肉痠痛（0/1）

### Stage 1: Covid vs non-Covid

| Label | Define |
| --- | --- |
| covid | COVID-19 |
| non-Covid | healthy_cough, upper_infection, lower_infection, obstructive_disease |

### Stage 2: Healthy vs Symptomatic

| Label | Define |
| --- | --- |
| healthy | healthy_cough |
| symptomatic | upper_infection, lower_infection, obstructive_disease |

---

## Three stage

### Stage 1: Covid vs non-Covid

| Label | Define |
| --- | --- |
| covid | COVID-19 |
| non-Covid | healthy_cough, upper_infection, lower_infection, obstructive_disease |

### Stage 2: Healthy vs Symptomatic

| Label | Define |
| --- | --- |
| healthy | healthy_cough |
| symptomatic | upper_infection, lower_infection, obstructive_disease |

### Stage 3: Upper infection vs Lower infection vs Obstructive disease

特徵向量：
- `emb_0` ~ `emb_511`：HeAR embedding（512維）
- `age`：年齡（數值）=> 516 維
- `gender_encoded`：性別（male=0, female=1, other=2, unknown=-1）=> 516 維
- `respiratory_condition`：慢性呼吸道疾病（0/1）=> 516 維
- `fever_muscle_pain`：發燒或肌肉痠痛（0/1）=> 516 維
- `cough_type_encoded`：咳嗽類型，wet=1 / dry=0 / unknown=-1 => 520 維
- `dyspnea`：呼吸困難，1/0/-1 => 520 維
- `wheezing`：喘鳴聲，1/0/-1 => 520 維
- `congestion`：鼻塞，1/0/-1 => 520 維

#### Version 1
特徵向量：516 維

| Stage | Model | 特徵向量|
| --- | --- | --- |
| 1 | LogisticRegression | 516 |
| 2 | LogisticRegression | 516 |
| 3 | LogisticRegression | 516 |

#### Version 2

| Stage | Model | 特徵向量|
| --- | --- | --- |
| 1 | LogisticRegression | 516 |
| 2 | LogisticRegression | 516 |
| 3 | pipe | 516 |

```
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('svc', SVC(kernel='rbf', class_weight='balanced',
                probability=True, random_state=42))
])
```

#### Version 3

| Stage | Model | 特徵向量|
| --- | --- | --- |
| 1 | LogisticRegression | 516 |
| 2 | LogisticRegression | 516 |
| 3 | pipe | 520 |

```
pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote',  SMOTE(k_neighbors=5, random_state=42)),
    ('pca',    PCA(n_components=80)),
    ('clf',    LogisticRegression(
                   class_weight='balanced',
                   max_iter=1000,
                   solver='lbfgs',
                   random_state=42
               ))
])
```

#### Version 4

| Stage | Model | 特徵向量|
| --- | --- | --- |
| 1 | pipe | 516 |
| 2 | pipe | 516 |
| 3 | pipe | 520 |

```
pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote',  SMOTE(k_neighbors=5, random_state=42)),
    ('pca',    PCA(n_components=80)),
    ('clf',    LogisticRegression(
                   class_weight='balanced',
                   max_iter=1000,
                   solver='lbfgs',
                   random_state=42
               ))
])
```
