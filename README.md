# 🛒 E-Commerce Customer Churn Prediction

A full end-to-end machine learning pipeline to predict whether an e-commerce customer will churn, covering data preprocessing, unsupervised customer profiling, supervised classification, hyperparameter tuning, ensemble modeling, and model interpretability.

---

## 📌 Problem Statement

Customer churn is one of the most costly problems in e-commerce. This project builds a predictive system that identifies at-risk customers using their behavioral, transactional, and engagement data — enabling businesses to intervene before losing them.

---

## 📁 Dataset

**File:** `ecommerce_customer_churn_dataset.csv`  
**Target:** `Churned` (binary: 1 = churned, 0 = retained)

### Features (24 total)

| Category | Features |
|---|---|
| Demographics | Age, Gender, Country, City |
| Engagement | Login_Frequency, Session_Duration_Avg, Pages_Per_Session, Social_Media_Engagement_Score, Mobile_App_Usage, Email_Open_Rate |
| Transaction | Total_Purchases, Average_Order_Value, Lifetime_Value, Days_Since_Last_Purchase, Credit_Balance |
| Behavior | Cart_Abandonment_Rate, Wishlist_Items, Discount_Usage_Rate, Returns_Rate, Customer_Service_Calls, Product_Reviews_Written, Payment_Method_Diversity |
| Meta | Membership_Years, Signup_Quarter |

---

## 🔧 Pipeline Overview

### 1. Exploratory Data Analysis
- Missing value heatmap
- Churn rate breakdown by `Signup_Quarter` and `Gender`
- Outlier detection using **IQR**, **Z-score**, and **IsolationForest**

### 2. Preprocessing
- **KNN Imputer** for correlated engagement features (`Social_Media_Engagement_Score`, `Mobile_App_Usage`, `Email_Open_Rate`, `Session_Duration_Avg`, `Pages_Per_Session`)
- **Outlier capping** based on IQR bounds
- **RobustScaler** for numerical features (chosen over StandardScaler/MinMaxScaler after visual comparison)
- **Label Encoding** for categorical columns
- **PCA** for variance analysis

### 3. Unsupervised Customer Profiling (Clustering)
Clustered customers to discover natural behavioral segments before classification.

| Algorithm | Metric Used |
|---|---|
| K-Means | Elbow Method + Silhouette Score |
| Gaussian Mixture Models (GMM) | BIC + AIC |
| Agglomerative Clustering | Dendrogram |
| DBSCAN | k-NN distance plot for eps |

**Winner: K-Means (k=3)**
- Silhouette Score: **0.906**
- Lowest Davies-Bouldin Index
- Highest Calinski-Harabasz Score

### 4. Baseline Classification (5-Fold Stratified CV)
Evaluated with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC:
- Logistic Regression (L1 + L2)
- Decision Tree
- Random Forest

### 5. Advanced Boosting Models with Optuna Tuning
Compared **XGBoost**, **LightGBM**, and **CatBoost** across two class imbalance strategies:

| Strategy | Method |
|---|---|
| Oversampling | **SMOTE** |
| Cost-sensitive | **Class Weight** (`scale_pos_weight`) |

Hyperparameters tuned via **Optuna** (Bayesian optimization) for each model × strategy combination.

### 6. Ensemble Methods
- **VotingClassifier** (soft voting across LGB + XGB + CatBoost)
- **StackingClassifier** (meta-learner: Logistic Regression)

**Best Single Model: LightGBM**
```
n_estimators = 500 | learning_rate = 0.05 | num_leaves = 31
```

---

## 🔍 Model Interpretability

| Method | Scope |
|---|---|
| **Feature Importance** (LightGBM) | Global — top 10 churn drivers |
| **SHAP** (TreeExplainer) | Global — feature impact distribution |
| **LIME** (LimeTabularExplainer) | Local — per-customer explanation |
| **Partial Dependence Plots** | Global — how top 6 features influence churn probability |

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Preprocessing | `scikit-learn` (KNNImputer, RobustScaler, PCA) |
| Imbalance | `imbalanced-learn` (SMOTE) |
| Models | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| Tuning | `optuna` |
| Clustering | `scikit-learn` (KMeans, DBSCAN, AgglomerativeClustering), `scipy` |
| Explainability | `shap`, `lime` |
| Visualization | `matplotlib`, `seaborn` |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Shishir-Srivastava2005/Churn_Prediction.git
cd Churn_Prediction
pip install -r requirements.txt
jupyter notebook Churn_prediction.ipynb
```

---

## 📂 Project Structure

```
Churn_Prediction/
├── Churn_prediction.ipynb   # Full pipeline notebook
├── ecommerce_customer_churn_dataset.csv
└── README.md
```

---

## 👤 Author

**Shishir Srivastava**  
B.Tech CSE | GTBIT, GGSIPU  
[GitHub](https://github.com/Shishir-Srivastava2005) · [LinkedIn](https://linkedin.com/in/)
