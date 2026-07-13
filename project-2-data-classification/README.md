# Project 2 — Data Classification Using AI 🌸

**Goal:** Build a basic classification model using a small dataset (supervised learning).

We don't write the rules — we provide history (labeled data), and the machine derives
the logic. This project trains a K-Nearest Neighbors classifier on the classic Iris
benchmark and validates it properly.

## The pipeline (IPO framework)

| Phase | What happens |
|---|---|
| **Input** | Iris dataset: 150 balanced samples, 3 classes, 4 features. `StandardScaler` normalizes features to mean 0 / variance 1 so no dimension dominates the distance math. |
| **Process** | Shuffled, stratified **80/20 train-test split** (shuffle removes order bias; the test set stays locked away). **KNN** classifies by the majority vote of the K closest neighbors — "similar things exist in close proximity." |
| **Output** | **Confusion matrix + F1 score**, not just accuracy — in imbalanced data, accuracy is a mirage. Also saves an **elbow plot** showing how K was tuned (K=1 overfits noise, K too large underfits). |

## Run it

```bash
python3 classification.py
```

Outputs to the terminal (accuracy, F1, confusion matrix, per-class precision/recall)
and saves two images:

- `elbow_plot.png` — error rate vs. K, used to justify K=5
- `confusion_matrix.png` — TP/FP/FN/TN breakdown per class

## The scikit-learn workflow

```python
model = KNeighborsClassifier(n_neighbors=5)   # INSTANTIATE (build the frame)
model.fit(X_train, y_train)                   # FIT (memorize the map)
predictions = model.predict(X_test)           # PREDICT (apply logic)
```

## Key skills demonstrated

Data handling, feature scaling, train/test methodology, supervised learning basics,
model training, and honest evaluation (confusion matrix, precision/recall trade-offs,
F1 as the harmonic mean).
