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

🔬 Pipeline Breakdown
---------------------

**Part 1 — Exploratory Data Analysis**  

*   Missing value heatmap across 25 features (5–20% MAR & MNAR patterns)
    
*   Outlier detection via **IQR**, **Z-score**, and **Isolation Forest** (5% contamination)
    
*   Correlation matrix + **VIF analysis** to detect multicollinearity
    
*   KDE plots per feature stratified by churn label
    
*   Churn rate analysis by signup quarter, gender, and customer service calls
    
*   *   High cart abandonment + low session duration → strong churn signal
        
    *   3+ customer service calls correlates with frustration-driven churn
        
    *   High social media engagement but low mobile app usage indicates platform mismatch
        

**Part 2 — Preprocessing & Feature Engineering**  

**3 Imputation Strategies:**

StrategyFeaturesReasonZero fill (domain-based)Customer\_Service\_Calls, Returns\_Rate, Wishlist\_Items, Product\_Reviews\_WrittenNaN = no action takenKNN Imputer (k=5)Social\_Media\_Engagement\_Score, Mobile\_App\_Usage, Email\_Open\_Rate, Session\_Duration\_Avg, Pages\_Per\_SessionCorrelated engagement featuresIterativeImputer + LinearRegressionCredit\_Balance, Age, Discount\_Usage\_RateMNAR pattern — model-based is appropriate

**10+ Engineered Features:**

*   Recency\_Score, Frequency\_Score, Monetary\_Score → **RFM scoring** (quintile-based)
    
*   Frustration\_Score = Customer\_Service\_Calls × Returns\_Rate
    
*   Digital\_Engagement\_Score = Social + Mobile + Email scores
    
*   Pages\_per\_Minute = interaction rate intensity
    
*   Purchase\_vs\_Wishlist = conversion tendency
    
*   Customer\_Lifecycle\_Stage bins: New → Established → Loyal → Veteran
    
*   Age\_Group binning + Purchases\_per\_Login ratio
    

**Scaling:** RobustScaler applied post-outlier capping (1st–99th percentile) to handle skewed distributions.

**Part 3 — Customer Segmentation (Unsupervised)**  

**Dimensionality Reduction:**

*   PCA → 3 components (for clustering input)
    
*   t-SNE 2D on 5,000-sample subset for visualization
    
*   UMAP 3D on 5,000-sample subset — compared against t-SNE
    

**4 Clustering Algorithms Benchmarked:**

AlgorithmSilhouette ↑Davies-Bouldin ↓Calinski-Harabasz ↑**K-Means (k=3)0.906LowestHighest**GMM (k=3)ModerateModerateModerateAgglomerative (k=3)ModerateModerateModerateDBSCAN———

**Winner: K-Means (k=3)** — clearly dominant across all three metrics.

**3 Identified Customer Segments:**

*   High-Value Loyal → Low churn risk, high LTV
    
*   At-Risk Bargain Hunters → Medium churn, high discount dependency
    
*   Disengaged → Low activity, highest churn rate
    

**Part 4 — Churn Prediction (Supervised)**  

**Baseline Models (5-Fold Stratified CV):**

*   Logistic Regression (L1 + L2 regularization)
    
*   Decision Tree
    
*   Random Forest
    

**Advanced Models + Hyperparameter Tuning:**

*   XGBoost, LightGBM, CatBoost
    
*   Optuna Bayesian Optimization — **50 trials per study**, two studies (SMOTE & Class-Weight)
    

**Class Imbalance Handling (compared):**

*   SMOTE oversampling on training set
    
*   scale\_pos\_weight class weighting
    

**Ensemble Methods Built:**

MethodComponentsHard VotingLightGBM + XGBoost + CatBoostSoft VotingLightGBM + XGBoost + CatBoostStackingLGBM + RandomForest → LogisticRegression meta-learner

**Best Result: ROC-AUC = 0.9277** (single model, class-weighted)

**Part 5 — Model Interpretation & Business Insights**  

*   **SHAP TreeExplainer** → global summary plot + top 10 churn drivers (LightGBM)
    
*   **LIME** → local explanations for 3 individual cases: True Positive, False Positive, False Negative
    
*   **Partial Dependence Plots** → non-linear churn relationship for top 6 features
    
*   Business recommendations per segment with estimated revenue impact of 10% churn reduction

---

Tech Stack

<table>
  <tr>
    <th>Category</th>
    <th>Libraries</th>
  </tr>
  <tr>
    <td>Data</td>
    <td>pandas, numpy</td>
  </tr>
  <tr>
    <td>Visualization</td>
    <td>matplotlib, seaborn, plotly</td>
  </tr>
  <tr>
    <td>Preprocessing</td>
    <td>scikit-learn (StandardScaler, RobustScaler, KNNImputer, IterativeImputer), imbalanced-learn (SMOTE)</td>
  </tr>
  <tr>
    <td>Unsupervised</td>
    <td>KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture, PCA, t-SNE, UMAP</td>
  </tr>
  <tr>
    <td>Supervised</td>
    <td>XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression</td>
  </tr>
  <tr>
    <td>Ensembles</td>
    <td>VotingClassifier, StackingClassifier</td>
  </tr>
  <tr>
    <td>Tuning</td>
    <td>Optuna (Bayesian Optimization)</td>
  </tr>
  <tr>
    <td>Interpretability</td>
    <td>SHAP, LIME, Partial Dependence Plots</td>
  </tr>
  <tr>
    <td>Statistics</td>
    <td>scipy, statsmodels (VIF)</td>
  </tr>
</table>

---
