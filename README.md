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

#### Version 1
特徵向量（516 維）：
- `emb_0` ~ `emb_511`：HeAR embedding（512維）
- `age`：年齡（數值，標準化）
- `gender`：性別（male=0, female=1, other=2）
- `respiratory_condition`：慢性呼吸道疾病（0/1）
- `fever_muscle_pain`：發燒或肌肉痠痛（0/1）

| Stage | Model |
| --- | --- |
| 1 | LogisticRegression |
| 2 | LogisticRegression |
| 3 | LogisticRegression |

#### Version 2
特徵向量（516 維）：
- `emb_0` ~ `emb_511`：HeAR embedding（512維）
- `age`：年齡（數值，標準化）
- `gender`：性別（male=0, female=1, other=2）
- `respiratory_condition`：慢性呼吸道疾病（0/1）
- `fever_muscle_pain`：發燒或肌肉痠痛（0/1）

| Stage | Model |
| --- | --- |
| 1 | LogisticRegression |
| 2 | LogisticRegression |
| 3 | SVM |
