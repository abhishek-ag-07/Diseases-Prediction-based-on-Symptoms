# ----------------------------------------------------------
# 🔥 Disease Prediction Model (RF + XGBoost + LightGBM Ensemble)
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("Loading dataset...")
df_comb = pd.read_csv(
r"C:\Users\LENOVO\OneDrive\Desktop\dummy\Disease-Prediction\Dataset\dis_sym_dataset_comb.csv")
df_norm = pd.read_csv(
r"C:\Users\LENOVO\OneDrive\Desktop\dummy\Disease-Prediction\Dataset\dis_sym_dataset_norm.csv")


# Split features & labels
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0]

symptoms_list = list(X.columns)
diseases_list = sorted(list(set(Y)))

print(f"Dataset shape: {df_comb.shape}")
print(f"Number of symptoms: {len(symptoms_list)}")
print(f"Number of diseases: {len(diseases_list)}")

print("\nSplitting for evaluation (90% train / 10% test)...")
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42
)

# ----------------------------------------------------------
# 🔥 MODELS
# ----------------------------------------------------------

print("\nInitializing models...")

model_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='multi:softprob',
    num_class=len(diseases_list),
    eval_metric="mlogloss"
)

model_lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=60,
    objective='multiclass',
    num_class=len(diseases_list)
)

# ----------------------------------------------------------
# 🔥 ENSEMBLE
# ----------------------------------------------------------

print("\nCreating Ensemble Voting Classifier...")

ensemble = VotingClassifier(
    estimators=[
        ('rf', model_rf),
        ('xgb', model_xgb),
        ('lgb', model_lgb)
    ],
    voting='soft'
)

# ----------------------------------------------------------
# 🔥 TRAINING
# ----------------------------------------------------------

print("\nTraining Ensemble Model (Random Forest + XGBoost + LightGBM)...")
ensemble.fit(X, Y)

# ----------------------------------------------------------
# 🔥 EVALUATION
# ----------------------------------------------------------

print("\nEvaluating model on test set...")
y_pred = ensemble.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

print(f"Model Test Accuracy: {accuracy:.2f}%")
print(f"Model F1-Score (weighted): {f1:.2f}%")

# 🔥 Classification Report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, digits=4, zero_division=0)
print(report)

with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Classification report saved as 'classification_report.txt'")

# ----------------------------------------------------------
# 🔥 CONFUSION MATRIX
# ----------------------------------------------------------
print("\nPlotting Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred, labels=diseases_list)
plt.figure(figsize=(18, 13))
sns.heatmap(cm, xticklabels=diseases_list, yticklabels=diseases_list, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved: confusion_matrix.png")
plt.close()

# ----------------------------------------------------------
# 🔥 FEATURE IMPORTANCE (Top 25)
# ----------------------------------------------------------
print("\nPlotting Feature Importance (Top 25)...")

# Extract trained RandomForest model from ensemble
rf_trained = ensemble.named_estimators_['rf']

# Get feature importance
rf_imp = rf_trained.feature_importances_
importance_df = pd.DataFrame({
    "symptom": symptoms_list,
    "importance": rf_imp
}).sort_values(by="importance", ascending=False).head(25)

plt.figure(figsize=(10, 8))
plt.barh(importance_df["symptom"], importance_df["importance"])
plt.gca().invert_yaxis()  # highest on top
plt.title("Top 25 Important Symptoms (Random Forest)")
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("Feature importance saved: feature_importance.png")

# ----------------------------------------------------------
# 🔥 SAVE MODEL & FILES
# ----------------------------------------------------------
print("\nSaving model and files...")

with open("model.pkl", "wb") as f:
    pickle.dump(ensemble, f)

with open("symptoms_list.json", "w") as f:
    json.dump(symptoms_list, f, indent=2)

with open("diseases_list.json", "w") as f:
    json.dump(diseases_list, f, indent=2)

df_norm.to_pickle("dataset_norm.pkl")

model_info = {
    "model_type": "Ensemble_RF_XGB_LGB",
    "test_accuracy": float(accuracy),
    "f1_score_weighted": float(f1),
    "trained_on": "full_dataset",
    "num_diseases": len(diseases_list),
    "num_symptoms": len(symptoms_list)
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("\n[SUCCESS] Ensemble model training completed!")
