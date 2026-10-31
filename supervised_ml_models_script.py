import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# ==========================================
# 1. CLASSIFICATION PIPELINE
# ==========================================
print("--- 1. CLASSIFICATION PIPELINE ---")
X_cls, y_cls = make_classification(n_samples=500, n_features=6, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

cls_models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

cls_results = []
for name, model in cls_models.items():
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, preds)
    prec = precision_score(y_test_c, preds)
    rec = recall_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds)
    cls_results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    })

df_cls = pd.DataFrame(cls_results)
print(df_cls.to_string(index=False))

# ==========================================
# 2. REGRESSION PIPELINE
# ==========================================
print("\n--- 2. REGRESSION PIPELINE ---")
X_reg, y_reg = make_regression(n_samples=500, n_features=6, noise=15.0, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)
reg_preds = reg_model.predict(X_test_r)

mse = mean_squared_error(y_test_r, reg_preds)
r2 = r2_score(y_test_r, reg_preds)

print(f"Linear Regression MSE: {mse:.4f}")
print(f"Linear Regression R2-Score: {r2:.4f}")

# Export classification comparison to CSV deliverable
df_cls.to_csv("model_comparison_results.csv", index=False)