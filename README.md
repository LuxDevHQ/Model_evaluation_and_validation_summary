# Model Evaluation & Validation Techniques

### Topic: Cross-validation and Performance Metrics

---

##  Summary

* Why **train/test split isn't enough**
* Learn **K-Fold Cross-Validation** for more robust model evaluation
* Recap of **classification and regression metrics**
* Understand **ROC curves, AUC**, and **Precision-Recall** curves
* Learn how to **detect and avoid overfitting/underfitting**

---

## 1. Why Train/Test Split Isn’t Enough

Splitting your data into a **training set** and a **test set** is a common approach — but it has **limitations**:

###  Problems with train/test split:

* High **variance** in performance depending on how the data was split
* Risk of **overfitting to the test set** if tuned repeatedly
* May not give a true picture of how well your model generalizes

---

###  Analogy: Studying for One Exam

> You study math all semester and then take **just one exam**.
> What if that exam happened to have questions on your weakest topic? Your performance would look poor even if you're generally good at math.

> Cross-validation is like taking **multiple exams** on different parts of the syllabus and averaging your results — **more fair** and **more reliable**.

---

## 2. Cross-Validation (CV)

###  What is Cross-Validation?

A technique to assess how your model will generalize to **unseen data** by training and testing it on **multiple subsets** of your dataset.

---

###  K-Fold Cross-Validation

1. Split data into **K equal parts** (folds)
2. Train the model on **K-1 folds**
3. Test the model on the **remaining fold**
4. Repeat this **K times**, using a different fold as test set each time
5. Average the performance scores

---

###  Analogy: Multiple Job Interviews

> Instead of evaluating a job candidate on one question, you ask them **5 different questions** in 5 interviews. This gives a **well-rounded** assessment.

---

###  K-Fold CV Code (Classification)

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Average Accuracy:", scores.mean())
```

---

## 3. Evaluation Metrics Recap

---

###  Classification Metrics

| Metric        | Description                                 | Use                             |
| ------------- | ------------------------------------------- | ------------------------------- |
| **Accuracy**  | % of correct predictions                    | Works when classes are balanced |
| **Precision** | Correct positives / All predicted positives | When false positives are costly |
| **Recall**    | Correct positives / All actual positives    | When false negatives are costly |
| **F1-Score**  | Harmonic mean of precision and recall       | Balances precision and recall   |
| **ROC-AUC**   | Area under ROC curve                        | Overall model quality           |

---

#### Confusion Matrix

|            | Predicted Yes  | Predicted No   |
| ---------- | -------------- | -------------- |
| Actual Yes | TP (True Pos)  | FN (False Neg) |
| Actual No  | FP (False Pos) | TN (True Neg)  |

---

####  ROC and AUC

* **ROC Curve**: Plots **True Positive Rate** vs **False Positive Rate**
* **AUC** (Area Under Curve): The **bigger**, the **better** (closer to 1)

---

####  ROC Code Example

```python
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

###  Regression Metrics

| Metric                                      | Description                                                                  |
| ------------------------------------------- | ---------------------------------------------------------------------------- |
| **MAE (Mean Absolute Error)**               | Average absolute difference between predicted and actual values              |
| **MSE (Mean Squared Error)**                | Like MAE but penalizes large errors more                                     |
| **RMSE**                                    | Square root of MSE — in same units as target                                 |
| **R² Score (Coefficient of Determination)** | % of variance explained by the model (1 = perfect, 0 = no explanatory power) |

---

###  Regression Evaluation Code

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(n_samples=100, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## 4. Overfitting vs Underfitting

| Term             | Meaning              | Symptom                       | Fix                               |
| ---------------- | -------------------- | ----------------------------- | --------------------------------- |
| **Underfitting** | Model is too simple  | Poor train/test accuracy      | Use more features, complex model  |
| **Overfitting**  | Model is too complex | High train, low test accuracy | Use regularization, CV, more data |

---

###  Analogy: Studying for Exams

> **Underfitting**: Didn’t study enough — can’t even pass mock tests.
> **Overfitting**: Memorized the practice exam — aces the mock, fails real one.
> **Good model**: Understood concepts — performs well on both.

---

## 5. Summary Table

| Topic            | Description                    |
| ---------------- | ------------------------------ |
| Train/Test Split | Basic, but limited             |
| K-Fold CV        | Reliable model validation      |
| ROC-AUC          | Model's classification power   |
| MAE, MSE, R²     | Regression performance metrics |
| Overfitting      | Model too complex              |
| Underfitting     | Model too simple               |

---

## 6. Final Analogy Recap

| Analogy                    | Concept                          |
| -------------------------- | -------------------------------- |
| Multiple Exams             | K-Fold Cross-Validation          |
| Hiring Interviews          | Evaluation from different angles |
| Over-studying / Memorizing | Overfitting                      |
| Too little study           | Underfitting                     |

---

