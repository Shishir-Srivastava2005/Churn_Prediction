# E-Commerce Customer Segmentation & Churn Prediction

## Project Overview
This project provides a comprehensive data science solution for a large-scale e-commerce company. The goal is to segment customers based on their behavior patterns and predict the likelihood of churn for each segment. This allows the business team to implement data-driven retention strategies.

The project addresses real-world data challenges, including:
* **Hidden Anomalies:** Identifying and correcting illogical data points (e.g., negative purchase counts or age outliers).
* **Class Imbalance:** Managing datasets where churned users are the minority.
* **Feature Engineering:** Extracting value from 25+ demographic, behavioral, and financial features.

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

