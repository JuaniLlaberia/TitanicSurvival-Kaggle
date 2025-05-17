# 🚢 Titanic Survival Prediction with XGBoost

This repository contains a complete machine learning pipeline using the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic) to predict survival using XGBoost. The notebook demonstrates data cleaning, feature engineering, model building, hyperparameter tuning, and result visualization.

🏆 Bronze medal

## 📁 Dataset

The data used in this project is from the Titanic competition available at: [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)

Files:
- `train.csv`: Training data
- `test.csv`: Test data

## ⚙️ Key Features of the Pipeline

### ✅ Preprocessing and Feature Engineering
- **Status Extraction** from passenger names (e.g., Mr., Miss, etc.)
- **Cabin Category Simplification**
- **Age Imputation** based on social status (e.g., Mr., Mrs., Master, etc.)
- **Embarked Filling** using most common embarkation port
- **Feature Encoding** with `pd.get_dummies()` for categorical variables
- **Child Classification** based on age (≤12 = child)
- **Outlier Removal** using Z-score thresholding on fare
- **Normalization** with Min-Max scaling
- **Highly Correlated Feature Removal**

### 📊 Data Exploration
- Class distribution for key variables (`Sex`, `Pclass`, `Embarked`)
- Visualizations for outlier handling and feature importance

### 🧠 Modeling
- **Model:** `XGBClassifier` from `xgboost`
- **Baseline Model:** Trained without hyperparameters
- **Hyperparameter Tuning:** Done with `RandomizedSearchCV` (commented block)
- **Final Model:** Trained with best parameters from tuning

### 📈 Evaluation
- Accuracy
- F1 Score
- Recall Score
- Feature Importance Plot

### 📤 Submission
- Final predictions are saved to `submission.csv` for Kaggle submission.

## 📌 Requirements

Install necessary libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
