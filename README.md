# HR Employee Flight Risk Prediction 🚀

An AI-driven predictive system built to combat voluntary attrition by identifying high-risk employees before they exit. Using a composite ensemble learning model, it translates raw demographic and engagement signals into actionable retention interventions.

---

## 🌟 Project Overview

Employee churn entails substantial replacement costs and operational drag. This repository centralizes a modern end-to-end ML framework that analyzes historical engagement data, counters class imbalance via oversampling, and predicts flight probability thresholds.

### Key Objectives
1. **Class-Balanced Optimization**: Employs `SMOTE` to rectify severe real-world attrition imbalances (~16% minority rate).
2. **Dynamic Soft-Voting Ensemble**: Fuses Gradient Boosting (`XGBoost`, `sklearn`), `RandomForest`, and calibrated `LogisticRegression` to maximize stable generalization.
3. **Strategic Interventions**: Distills abstract probability metrics into tangible color-coded Excel rosters tailored for Direct Management usage.

---

## 📂 Repository Roadmap

```plaintext
HR_Employee_Flight_Prediction/
├── HR_Project.ipynb            # Core source-of-truth (Training, Eval, Reports)
├── train_hr_data.csv           # Static training reference data
├── validate_hr_data.csv        # Blind validation testbed dataset
├── retention_risk_report.xlsx  # High-level colorized actionable summary
├── requirements.txt            # Managed environment dependencies
└── README.md                   # System documentation
```

---

## ⚙️ Core Architecture & Pipeline

The methodology contained within the primary [**HR_Project.ipynb**](./HR_Project.ipynb) integrates the following cycles:

### 1. Smart Feature Synthesis
Aggregates static demographics into behavioral indicators:
*   **IncomePerYear**: Evaluates effective yield relative to loyalty.
*   **WorkLifeStress**: Interaction of commute distance, overtime, and involvement.
*   **TenureRatio**: Analyzes managerial tenure dynamics.

### 2. Predictive Logic
*   **Oversampling**: Synthetic Minority Over-sampling Technique (`SMOTE`) invoked post-normalization to generate unbiased feature boundaries.
*   **Soft-Weight Voting**: Weighted combinations favor robust tree-based predictors while retaining the stable convergence of linear priors (`weights=[3, 1, 3, 2]`).
*   **Intervention Threshold (`0.35`)**: Calibrated aggressively to flag early attrition signals, maximizing actionable Recall over raw Precision conservatism.

### 3. Automated HR Reporting
Outputs static results to `retention_risk_report.xlsx`, enforcing real-time color-formatting criteria:
*   🔴 **High Risk** (>60% likelihood)
*   🟡 **Medium Risk** (35% - 60% likelihood)
*   🟢 **Low Risk** (<35% likelihood)

---

## 🚀 Setup & Deployment

### Local Prerequisites
Ensure `Python 3.8+` is operational locally. Run standard dependency recovery:

```bash
pip install -r requirements.txt
```

### Execution
Simply launch Jupyter and execute the monolithic flow in `HR_Project.ipynb` to auto-generate performance metrics and the refreshed `retention_risk_report.xlsx` downstream.

```bash
jupyter notebook HR_Project.ipynb
```

---

## 📊 Reliability Assessment
*   **Validation Metric**: 5-Fold Stratified Cross-Validation for absolute stability across divergent distributions.
*   **Metrics Evaluated**: Explicit output of Precision, Recall, and composite F1-Score, ensuring the model successfully tracks minority churn cohorts rather than defaulting to majority accuracy inflation.
