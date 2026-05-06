"""
AI_stats_lab.py

Instructor Solution
Lab: Training and Evaluating Classification Models
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Question 1: Confusion Matrix, Metrics, and Threshold Effects
# ============================================================

def confusion_matrix_counts(y_true, y_pred):
    """
    Return:
        (TP, FP, FN, TN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return int(TP), int(FP), int(FN), int(TN)


def classification_metrics(y_true, y_pred):
    """
    Return:
        {
            "recall": value,
            "fallout": value,
            "precision": value,
            "accuracy": value
        }
    """
    TP, FP, FN, TN = confusion_matrix_counts(y_true, y_pred)

    recall_denominator = TP + FN
    fallout_denominator = FP + TN
    precision_denominator = TP + FP
    total = TP + FP + FN + TN

    recall = TP / recall_denominator if recall_denominator != 0 else 0.0
    fallout = FP / fallout_denominator if fallout_denominator != 0 else 0.0
    precision = TP / precision_denominator if precision_denominator != 0 else 0.0
    accuracy = (TP + TN) / total if total != 0 else 0.0

    return {
        "recall": recall,
        "fallout": fallout,
        "precision": precision,
        "accuracy": accuracy
    }


def apply_threshold(scores, threshold):
    """
    Return:
        NumPy array of 0/1 predictions.
    """
    scores = np.array(scores)
    return (scores >= threshold).astype(int)


def threshold_metrics_analysis(y_true, scores, thresholds):
    """
    Return a list of dictionaries, one per threshold.
    """
    results = []

    for threshold in thresholds:
        y_pred = apply_threshold(scores, threshold)
        metrics = classification_metrics(y_true, y_pred)

        results.append({
            "threshold": threshold,
            "recall": metrics["recall"],
            "fallout": metrics["fallout"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"]
        })

    return results


# ============================================================
# Question 2: Train Two Classifiers and Evaluate Them
# ============================================================

def train_two_classifiers(X_train, y_train):
    """
    Train logistic regression and decision tree classifiers.
    """
    logistic_model = LogisticRegression(max_iter=1000)
    decision_tree_model = DecisionTreeClassifier(random_state=0)

    logistic_model.fit(X_train, y_train)
    decision_tree_model.fit(X_train, y_train)

    return {
        "logistic_regression": logistic_model,
        "decision_tree": decision_tree_model
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained classifier.
    """
    scores = model.predict_proba(X_test)[:, 1]
    y_pred = apply_threshold(scores, threshold)

    TP, FP, FN, TN = confusion_matrix_counts(y_test, y_pred)
    metrics = classification_metrics(y_test, y_pred)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall": metrics["recall"],
        "fallout": metrics["fallout"],
        "precision": metrics["precision"],
        "accuracy": metrics["accuracy"]
    }


def compare_classifiers(X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Train and evaluate both classifiers.
    """
    models = train_two_classifiers(X_train, y_train)

    logistic_results = evaluate_classifier(
        models["logistic_regression"],
        X_test,
        y_test,
        threshold
    )

    decision_tree_results = evaluate_classifier(
        models["decision_tree"],
        X_test,
        y_test,
        threshold
    )

    return {
        "logistic_regression": logistic_results,
        "decision_tree": decision_tree_results
    }


if __name__ == "__main__":
    print("Instructor solution loaded.")
