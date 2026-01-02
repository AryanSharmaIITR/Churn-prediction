# Customer Churn Prediction

**End-to-End Machine Learning Project | Classification | Business Analytics**

## Business Problem
Customer churn directly impacts revenue and growth. The objective of this project is to identify customers likely to churn so that businesses can take proactive retention actions.

This project demonstrates an end-to-end machine learning workflow, from data analysis to interpretable modeling, with a strong focus on real-world decision-making.

## Key Highlights (Recruiter Summary)
- Built a complete ML pipeline: EDA → Feature Engineering → Modeling → Evaluation
- Used interpretable ML (Logistic Regression) to explain churn drivers
- Applied statistical outlier handling and feature distribution analysis
- Validated model using train-test performance comparison
- Emphasized business interpretability, not just accuracy

## Tech Stack
- **Programming:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **ML Type:** Supervised Classification
- **Environment:** Jupyter Notebook

## Dataset Overview
- **Source:** Customer churn dataset (`Churn_prediction.csv`)
- **Target Variable:** Churn (Yes / No)
- **Data Size:** ~7K customer records
- **Features:**
  - *Numerical:* tenure, MonthlyCharges, TotalCharges
  - *Categorical:* service subscriptions, contracts, payment methods

## Exploratory Data Analysis (EDA)
Key insights generated through EDA:
- Identified high-risk churn segments based on tenure and charges
- Used IQR-based outlier detection to stabilize model training
- Visualized feature distributions across churn classes
- Compared churn vs non-churn customers using boxplots and statistics

EDA was treated as a decision-making tool, not a cosmetic step.

## Feature Engineering
- Removed non-informative identifiers (`customerID`)
- Encoded categorical variables
- Selected churn-sensitive numerical features
- Prepared clean, model-ready datasets

## Modeling Strategy
**Model Used:** Logistic Regression  
**Reason:**
- Strong baseline for classification
- High interpretability for business stakeholders
- Coefficient-based feature importance

## Model Evaluation
Evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Compared training vs test performance to check overfitting

Produced consistent results, indicating good generalization.

## Interpretability & Business Insights
Extracted model coefficients to identify:
- Features that increase churn probability
- Features that reduce churn risk

Enables actionable insights such as:
- Identifying at-risk customers early
- Designing targeted retention strategies

This makes the model deployable in real business environments, not just academic.

## Project Structure
├── data/
│ └── Churn_prediction.csv
├── EDA.ipynb # Data analysis & insights
├── Model.ipynb # Modeling & evaluation
├── README.md

text

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
## How to Run
Run notebooks in order:
- EDA.ipynb
- Model.ipynb

## Results Summary
Successfully built a predictive churn model

Identified key churn drivers using interpretable ML

Demonstrated strong ML fundamentals and clean workflow

## Future Enhancements
- Handle class imbalance using SMOTE / class weights
- Experiment with tree-based models (XGBoost, Random Forest)
- Add ROC-AUC and Precision-Recall curves
- Package as a production ML pipeline (MLflow / DVC)

## About the Author
**Aryan Sharma**  
Aspiring Machine Learning Engineer  
Strong interest in data-driven decision systems and production-ready ML