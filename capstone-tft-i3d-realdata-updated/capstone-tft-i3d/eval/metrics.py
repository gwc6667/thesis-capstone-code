import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def classification_metrics(y_true, y_pred, average='weighted'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
