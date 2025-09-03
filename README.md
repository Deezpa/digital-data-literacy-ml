
![License](https://img.shields.io/badge/license-MIT-informational)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-initial--release-brightgreen)

# Digital Data Literacy Program (AIWC/Ujjawal) — ML Extension

**Tagline:** Measuring and improving digital & financial literacy outcomes using machine learning, with privacy-preserving analytics and fairness audits.

## Overview
This repository extends the Ujjawal Women Association's Digital Data Literacy Program into an ML-ready project. It provides reproducible pipelines to (i) ingest anonymized training/assessment data, (ii) engineer features for learning outcomes, (iii) predict retention and mastery, and (iv) generate actionable cohort insights and dashboards.

- **PI:** Dr Deepa Shukla (ORCID: 0000-0003-3016-1633)
- **Impact:** 5,000+ women trained across India
- **Ethics:** De-identified, consented analytics; bias and fairness checks documented in `reports/`.

## Data Schema (suggested)
- `participant_id` (hash), `age_band`, `region`, `literacy_level_baseline`, `module_hours`, `assessment_pre`, `assessment_post`, `followup_90d`, `dropout_flag`, `device_access`, `net_availability`, `income_band`.

## ML Tasks
- **Binary:** dropout prediction; follow-up completion
- **Regression:** improvement score (post - pre)
- **Uplift:** treatment effect of module variants
- **Clustering:** learner personas

## Getting Started
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/ingest/load_data.py
```

## Reproducibility
- Version datasets via Git tags and releases.
- Use `CITATION.cff` for citation; consider DataCite DOI via Harvard Dataverse.

## Responsible AI
- Differential privacy options for reports; group fairness metrics; PII handling policies in `data/README.md`.


## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/ingest/synth_generate.py --n 1000 --out data/raw/ddl_synth.csv
python src/features/make_features.py --input data/raw/ddl_synth.csv --out data/processed/ddl_features.csv
python src/models/train_rf.py  # expects data/processed with 'target' column
python src/evaluation/fairness.py --input data/processed/ddl_features.csv --target target --group region
python src/visualization/shap_summary.py --model models/rf.joblib --data data/processed/ddl_features.csv --target target
```

## API (optional)
```bash
uvicorn src.serve:app --reload --port 8000
# POST /predict with JSON: {"feature1": 1, "feature2": "x", ...}
```
