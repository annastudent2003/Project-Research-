import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where this script is
DATA_PATH = os.path.join(BASE_DIR, "../data/heart.xlsx")  # dataset in data/
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs")         # outputs folder

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 1. LOAD DATASET
# ============================

print("Loading dataset...")
df = pd.read_excel(DATA_PATH)
print("Dataset Loaded Successfully!")
print(df.head())

# ============================
# 2. TABLES & PLOTS FOR IEEE REPORT
# ============================

# Table 1: Dataset Overview
print("\nTABLE 1: DATASET OVERVIEW")
print(df.info())
print("\nTarget Distribution:")
print(df['target'].value_counts())

# Save class distribution plot
plt.figure(figsize=(5,4))
sns.countplot(x=df['target'])
plt.title("Class Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
plt.close()

# Table 2: Feature Description Table (basic template)
df.describe().to_csv(os.path.join(OUTPUT_DIR, "feature_description.csv"))

# Correlation Heatmap
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# ============================
# 3. PREPROCESSING
# ============================

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed datasets
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_train.csv"), index=False
)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_test.csv"), index=False
)

print("\nPreprocessing completed & CSVs saved.")

# ============================
# 4. LOGISTIC REGRESSION
# ============================

lr = LogisticRegression(max_iter=200)
lr.fit(X_train_scaled, y_train)

coefficients = lr.coef_[0]
feature_names = X.columns

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
})

# Normalized weights (clinical weights)
coef_df["Normalized Weight"] = abs(coef_df["Coefficient"]) / sum(abs(coef_df["Coefficient"]))

# Interpretation
coef_df["Interpretation"] = np.where(
    coef_df["Coefficient"] > 0,
    "Higher value increases heart failure risk",
    "Higher value decreases heart failure risk"
)

# Save coefficients
coef_df.to_csv(os.path.join(OUTPUT_DIR, "LR_coefficients.csv"), index=False)

print("\nLogistic Regression training completed.")
print("Coefficients saved to outputs/LR_coefficients.csv")

print("\nAll tasks completed successfully!")
