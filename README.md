# Digital Data Literacy Program (AIWC/Ujjawal) — ML Extension

[![DOI](https://img.shields.io/badge/DOI-10.7910%2FDVN%2FEGAIKO-blue?logo=dataverse)](https://doi.org/10.7910/DVN/EGAIKO)
[![GitHub Release](https://img.shields.io/github/v/release/Deezpa/digital-data-literacy-ml)](https://github.com/Deezpa/digital-data-literacy-ml/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Tagline**: Measuring and improving digital & financial literacy outcomes using machine learning, with privacy-preserving analytics and fairness audits.

---

## 🔎 Project Overview
This repository extends the Ujjawal Women Association's **Digital Data Literacy Program** into an ML-ready project. It provides reproducible pipelines to:

1. Ingest anonymized training/assessment data  
2. Perform feature engineering for learning outcomes  
3. Predict retention and mastery  
4. Generate actionable cohort insights and dashboards  

- **PI**: Dr. Deepa Shukla (ORCID: [0000-0003-3016-1633](https://orcid.org/0000-0003-3016-1633))  
- **Impact**: 5,000+ women trained across India  
- **Ethics**: De-identified, consented analytics; bias and fairness checks documented in `reports/`

---

## 📊 Reports & Documentation
- 📑 [Model Card](reports/MODEL_CARD.md)  
- ⚖️ [Fairness Report](reports/FAIRNESS_RUN.md)  
- 🖼️ [SHAP Summary Plot](reports/figures/shap_summary.png)  
- 📦 [Release v0.2.2 Artifacts](https://github.com/Deezpa/digital-data-literacy-ml/releases/tag/v0.2.2)

---

## 🗂️ Data Schema (suggested)
`participant_id (hash), age_band, region, literacy_level_baseline, module_hours, assessment_pre, assessment_post, followup_90d, dropout_flag, device_access, net_availability, income_band`

---

## 🤖 ML Tasks
- **Binary classification**: dropout prediction; follow-up completion  
- **Regression**: learning gain score (post - pre)  
- **Uplift**: treatment effect of module variants  
- **Clustering**: learner personas  

---

## ⚙️ Getting Started
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/ingest/load_data.py

