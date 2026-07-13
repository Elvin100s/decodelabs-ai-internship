"""
Project 2 — Data Classification Using AI (DecodeLabs Industrial Training Kit)

Supervised learning pipeline on the Iris benchmark:
  INPUT   -> load Iris dataset, StandardScaler feature scaling
  PROCESS -> shuffled 80/20 train-test split, K-Nearest Neighbors
  OUTPUT  -> confusion matrix, F1 score, classification report + elbow plot for K
"""

import matplotlib

matplotlib.use("Agg")  # save plots to file, no display needed

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def main() -> None:
    # ---------- INPUT: load & understand the dataset ----------
    iris = load_iris()
    X, y = iris.data, iris.target
    print("=== Raw material: the Iris benchmark ===")
    print(f"Samples: {X.shape[0]} (balanced) | Classes: {len(iris.target_names)} | Dimensions: {X.shape[1]}")
    print(f"Features: {iris.feature_names}")
    print(f"Classes:  {list(iris.target_names)}\n")

    # Structural integrity: shuffle & split BEFORE fitting the scaler,
    # so no information from the test set leaks into training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train-test split: {len(X_train)} train / {len(X_test)} test (80/20, shuffled)\n")

    # The gatekeeper rule: scaling (mean=0, variance=1) so no feature dominates.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------- Tuning the engine: choosing K (elbow method) ----------
    k_values = range(1, 26)
    error_rates = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        error_rates.append(1 - accuracy_score(y_test, knn.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, error_rates, marker="o")
    plt.title("Tuning the engine: choosing K (elbow method)")
    plt.xlabel("K value")
    plt.ylabel("Error rate")
    plt.grid(True)
    plt.savefig("elbow_plot.png", dpi=120, bbox_inches="tight")
    print("Elbow plot saved to elbow_plot.png")

    # ---------- PROCESS: instantiate, fit, predict ----------
    model = KNeighborsClassifier(n_neighbors=5)  # INSTANTIATE (build the frame)
    model.fit(X_train, y_train)                  # FIT (memorize the map)
    predictions = model.predict(X_test)          # PREDICT (apply logic)

    # ---------- OUTPUT: validation beyond the "accuracy mirage" ----------
    print("\n=== Output validation (K=5) ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"F1 score (macro): {f1_score(y_test, predictions, average='macro'):.3f}\n")

    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm, "\n")

    print(classification_report(y_test, predictions, target_names=iris.target_names))

    ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap="Blues")
    plt.title("Confusion matrix — KNN (K=5) on Iris")
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Confusion matrix plot saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
