# CoughCare, Rag, Ollama
結合 RAG 技術與 Ollama 本地大型語言模型的智慧咳嗽分析Agent

## normal _or abnormal

### 正常 vs 異常

| Label | Define |
| --- | --- |
| normal | healthy_cough |
| abnormal | upper_infection, lower_infection, obstructive_disease, COVID-19 |

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
