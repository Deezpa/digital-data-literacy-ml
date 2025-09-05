# Model Card: Random Forest Credit Scoring (Synthetic Thin-File Data)

---

## 1. Model Details
- **Model type**: Random Forest Classifier
- **Implementation**: scikit-learn (v1.5.0)
- **Pipeline components**:
  - Synthetic data generator (`synth_generate.py`)
  - Feature engineering (`make_features.py`)
  - Model training (`train_rf.py`)
  - Fairness evaluation (`fairness.py`)
  - Explainability via SHAP (`shap_summary.py`)
- **Training data**: Synthetic dataset with 1,200 samples, 13 features
- **License**: MIT License (this repository)
- **Version**: v0.2.0

---

## 2. Intended Use
- **Primary purpose**: Demonstration of ML pipeline for credit scoring of 
thin-file consumers (borrowers with limited credit history).
- **Intended users**: 
  - Researchers in financial technology and Responsible AI
  - Educators teaching fairness-aware ML
  - Developers prototyping credit risk scoring tools
- **Decisions supported**: Binary classification of loan repayment 
likelihood (good/bad borrower).
- **Out-of-scope uses**: 
  - Production credit decisioning without further validation on real data
  - High-stakes lending without regulatory approval
  - Application to unrelated domains (e.g., healthcare risk prediction)

---

## 3. Training & Evaluation Data
- **Dataset**: Synthetic, generated with controlled statistical properties
- **Features include**: Age, income, employment status, repayment history 
proxies, digital/alternative features (simulated)
- **Target variable**: Loan repayment (binary: 0 = default, 1 = repay)
- **Train/test split**: 80% / 20%
- **Evaluation metrics**: 
  - AUC (Area Under ROC Curve): 1.00 (synthetic, not realistic)
  - Precision, Recall, F1-score: all 1.00 (synthetic, not realistic)

---

## 4. Quantitative Analyses
- **Performance on test set**:
  - Accuracy: 100%
  - Precision: 100%
  - Recall: 100%
- **Fairness evaluation** (see `FAIRNESS_RUN.md` for details):
  - Group variable tested: `region`
  - Outputs show no disparity on synthetic data
  - Note: Real-world bias may emerge on real datasets
- **Explainability**:
  - SHAP summary plot (`shap_summary.png`) indicates top contributing 
features

---

## 5. Ethical Considerations
- **Bias & Fairness**: Synthetic data eliminates real-world disparities; 
however, real data may amplify biases if not mitigated.
- **Transparency**: Code, data generation process, and metrics are fully 
open-source and reproducible.
- **Accountability**: This model is for research and educational purposes 
only; not to be used in real-world loan approvals.
- **Limitations**:
  - Unrealistic perfect scores due to synthetic data
  - Model not validated on real-world consumer credit data
  - Generalizability is untested beyond simulated environment

---

## 6. Caveats and Recommendations
- **Caveats**:
  - Synthetic dataset lacks real-world noise, adversarial behaviors, and 
reporting inconsistencies
  - Over-optimistic performance metrics (AUC = 1.00)
- **Recommendations**:
  - Apply pipeline to real-world, diverse credit datasets
  - Perform fairness audits across multiple protected attributes (gender, 
caste, income group)
  - Compare against baseline models (logistic regression, gradient 
boosting, neural networks)
  - Monitor drift and retrain periodically in real deployments

---

## 7. References
- Mitchell et al., “Model Cards for Model Reporting,” *FAT* 2019
- World Bank (2024). *Alternative Data in Credit Scoring: Policy 
Guidelines*
- Shukla, D. (2024). *A Survey of Machine Learning Algorithms in Credit 
Risk Assessment* (Journal of Electrical Systems)

---

## 8. Contact
- **Author**: Dr. Deepa Shukla  
- **Institution**: Jaipur National University, India  
- **Repository**: [GitHub – 
Deezpa/digital-data-literacy-ml](https://github.com/Deezpa/digital-data-literacy-ml)  
- **Dataset**: Published on [Harvard 
Dataverse](https://dataverse.harvard.edu/) (linked datasets)

---


