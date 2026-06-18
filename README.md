<h1 align="center"> Customer Churn Prediction</h1>
<p align="center">
  <b>End-to-End ML Pipeline · Customer Segmentation · Ensemble Modeling · Explainability</b>
</p>

<h2>Project Overview</h2>
A full production-grade ML pipeline built on a synthetic e-commerce dataset of **50,000 customers** across **25+ features** to:

1.  **Segment** customers using unsupervised clustering
    
2.  **Predict** 3-month churn with ensemble classifiers
    
3.  **Explain** model decisions using SHAP, LIME, and Partial Dependence Plots
    
4.  **Recommend** targeted retention strategies per customer segment
    

> **Highlights:** ROC-AUC of **0.9277**, Silhouette Score of **0.906**, Optuna-optimized ensemble with 100 total trials, and interpretability via SHAP + LIME.
---
Key Results

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ROC-AUC (Best Model)</td>
      <td><b>0.9277</b></td>
      <td>Class-weight tuned LightGBM/XGBoost/CatBoost</td>
    </tr>
    <tr>
      <td>Recall — Churn Class</td>
      <td><b>0.85</b></td>
      <td>Critical metric given class imbalance</td>
    </tr>
    <tr>
      <td>Precision — Churn Class</td>
      <td><b>0.84</b></td>
      <td>Balanced against recall</td>
    </tr>
    <tr>
      <td>F1-Score — Churn Class</td>
      <td><b>0.84</b></td>
      <td>Weighted, on 30% holdout test set</td>
    </tr>
    <tr>
      <td>Clustering Silhouette Score</td>
      <td><b>0.906</b></td>
      <td>K-Means (k=3), best across 4 algorithms</td>
    </tr>
    <tr>
      <td>Customer Segments Found</td>
      <td><b>3</b></td>
      <td>High-Value Loyal, At-Risk, Disengaged</td>
    </tr>
    <tr>
      <td>Optuna Trials</td>
      <td><b>100</b></td>
      <td>50 per study (SMOTE + Class-Weight)</td>
    </tr>
    <tr>
      <td>Features Engineered</td>
      <td><b>10+</b></td>
      <td>RFM scores, frustration score, digital engagement, etc.</td>
    </tr>
    <tr>
      <td>Dataset Size</td>
      <td><b>50,000</b></td>
      <td>With ~15% churn rate (class imbalanced)</td>
    </tr>
  </tbody>
</table>
---

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Techniques:** Unsupervised Clustering (K-Means), Random Forest/Ensemble Methods, Partial Dependence Plots (PDP)

## Workflow
1. **Exploratory Data Analysis (EDA):** Deep dive into 50,000 records to identify distributions, correlations, and data quality issues.
2. **Data Preprocessing:** * Implemented automated cleaning for extreme outliers.
    * Handled missing values in engagement scores and credit balances.
    * Encoded categorical variables (Gender, Country, City, Signup Quarter).
3. **Customer Segmentation:** Grouped customers into distinct personas using unsupervised learning to understand high-value vs. low-engagement behavior.
4. **Churn Prediction:** Built a classification model to predict user churn.
5. **Model Interpretation:** Used PDPs and feature importance to visualize the impact of variables like "Cart Abandonment Rate" and "Customer Service Calls" on churn.

## Key Insights
* **Churn Drivers:** Identified that specific behavioral triggers, such as high cart abandonment and low login frequency, are the strongest predictors of churn.
* **Segment Behavior:** High-value segments were found to be more sensitive to "Days Since Last Purchase," suggesting a need for automated re-engagement emails.

