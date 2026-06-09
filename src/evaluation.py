import pandas as pd

from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)


def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred),
        "predictions": y_pred,
    }
    
    return results

