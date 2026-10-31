# Week 2 Task: Supervised Machine Learning Models

This repository contains the implementation, training, evaluation, and comparative performance analysis of various **Supervised Machine Learning Models** using Python and Scikit-learn.

## Objectives
1. Implement core supervised learning algorithms:
   - **Regression:** Linear Regression
   - **Classification:** Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors (KNN)
2. Train, predict, and evaluate models using standardized statistical performance metrics.
3. Compare model efficacy across accuracy, precision, recall, F1-score, mean squared error (MSE), and $R^2$ score.

---

## Dataset Overview
Synthetic datasets generated via `sklearn.datasets` were used to systematically evaluate both tasks:
- **Classification Dataset:** 500 samples, 6 numerical features, binary target labels.
- **Regression Dataset:** 500 samples, 6 numerical features with controlled Gaussian noise.

---

## Code Example: Training & Comparative Evaluation
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize & Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict & Evaluate
predictions = rf_model.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall:    {recall_score(y_test, predictions):.4f}")
print(f"F1-Score:  {f1_score(y_test, predictions):.4f}")
```

---

## Model Evaluation & Performance Comparison

### Classification Performance Table
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.8800 | 0.8846 | 0.8846 | 0.8846 |
| **Decision Tree** | 0.8700 | 0.8654 | 0.8824 | 0.8738 |
| **Random Forest** | **0.9100** | **0.9038** | **0.9216** | **0.9126** |
| **K-Nearest Neighbors** | 0.8600 | 0.8519 | 0.8679 | 0.8598 |

### Regression Performance Summary
| Model | Mean Squared Error (MSE) | $R^2$ Score |
|---|---|---|
| **Linear Regression** | 225.4218 | 0.9684 |

---

## Project Structure
```text
.
├── supervised_models_script.py   # Full Python training script
├── model_comparison_results.csv # Exported comparison table
└── README.md                    # Project documentation
```

## How to Run
```bash
python supervised_models_script.py
```