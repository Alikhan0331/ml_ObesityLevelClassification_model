# Obesity Levels Classification – Multiclass ML Project

## Overview
This project predicts a person’s obesity level based on anthropometric data and lifestyle habits (nutrition, physical activity, smoking, etc.). It is a multiclass classification task using the Obesity Levels dataset from Kaggle.

## Data
- Source: Kaggle “Obesity Levels” dataset. 
- Target column: `NObeyesdad` – obesity level with multiple categories  
  (e.g., `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, ..., `Obesity_Type_III`). 
- Example feature groups:
  - Demographics: Gender.
  - Eating habits: number of main meals, consumption of high‑calorie food, and snacks.
  - Physical activity: time doing physical activity, sedentary time.
  - Health/lifestyle: family history of overweight, alcohol consumption, smoking. 

## Models
I compared several classic multiclass classifiers, each wrapped in a `Pipeline(preprocessor + model)`.

- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`) 
- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)  
- **Gradient Boosting Classifier** (`sklearn.ensemble.GradientBoostingClassifier`)  

## Evaluation
Main metric: **accuracy** – the fraction of correctly predicted obesity levels. 

LogisticRegression - 0.641378    
RandomForest       - 0.901975    
GradientBoosting   - 0.906551     

## Key Findings
- Proper one‑hot encoding of non‑binary categorical features significantly improves performance compared to naive integer encoding.
- Tree‑based models (RandomForest, GradientBoosting, XGBoost) outperform Logistic Regression on this mixed (numeric + categorical) dataset. 

## How to Run
1. Download the dataset from Kaggle (e.g.):  
   https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels or  
   https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset
2. Save the CSV as `obesity.csv`
3. Install dependencies and ipynb file
4. Run the ipynb file
