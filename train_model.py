"""
Train and evaluate the Food Safety Risk Assessment model.

IMPORTANT:
The included CSV is synthetic demonstration data. The target labels are
illustrative and are NOT FSSAI regulatory classifications.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "food_safety_demo.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["sample_id", "risk_label"])
y = df["risk_label"]

numeric_features = [
    "moisture_percent", "ph", "protein_g_100g", "fat_g_100g",
    "bacterial_count_cfu_g", "contaminant_level_mg_kg",
    "storage_temperature_c", "storage_days"
]
categorical_features = ["food_category"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
    )
}

results = {}
best_name = None
best_pipeline = None
best_f1 = -1

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, pos_label="At Risk"), 4),
        "recall": round(recall_score(y_test, pred, pos_label="At Risk"), 4),
        "f1_score": round(f1_score(y_test, pred, pos_label="At Risk"), 4)
    }
    results[name] = metrics

    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_name = name
        best_pipeline = pipeline

joblib.dump(best_pipeline, MODEL_DIR / "food_safety_model.pkl")

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "test_samples": len(X_test),
        "metrics": results
    }, f, indent=2)

pred = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, pred, labels=["Lower Risk", "At Risk"])

print("\nModel comparison:")
for name, metrics in results.items():
    print(name, metrics)

print("\nSelected model:", best_name)
print("\nClassification report:")
print(classification_report(y_test, pred))
print("\nConfusion matrix [Lower Risk, At Risk]:")
print(cm)
print("\nSaved:", MODEL_DIR / "food_safety_model.pkl")
