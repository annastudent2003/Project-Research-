# Project Research

This project predicts heart failure risk in a patient. Dataset used: Kaggle: Heart Disease UCI Dataset.

Models used: Logistic Regression, XGBoost, CatBoost, Random Forest.

* We preprocess the dataset, explore it (class imbalance, correlation heatmap etc) and then train Logistic Regression on it. Then we extract coefficients as clinical weights.
* We train three models on this same preprocessed dataset: catboost, randomforest, xgboost.
* Apply SHAP on catboost and xgboost to get the most important features. 
* We check for consistency. If consistent we use the important features from catboost for next step or else we either train the xgboost and catboost again to get better results or we preprocess the dataset again, in that case from starting all models here will train till we reach this step.
* Then we build the hybrid risk score formula. This is our novel work. We take LR coefficients, Catboost feature important features and calculate the score to assign a patient so that heart failure chances are counted.
* We run this formula on test dataset for comparison.

now we plot graphs to show which features have assigned high risk and why to some patients.

Then we use DiCE to show how and by what amount if reduce those features, we can lower the heart failure risk overall
