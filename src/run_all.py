import json
import sys
from pathlib import Path
from src.ingest import ingest
from src.transform import stage, curate

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from src.prepare import load_split, SEED

N_JOBS = 1
PARAMS = {
    "n_estimators": 276,
    "max_depth": 7,
    "learning_rate": 0.19781412884706517,
    "subsample": 0.8525778405097011,
    "colsample_bytree": 0.7012859229289896,
    "scale_pos_weight": 1.0,
}
BASELINE_F1 = 0.9865
BASELINE_AUC = 0.9991
BASELINE_CM = [[1882, 14], [16, 1096]]
LR_EXPECTED_CM = [[1853, 43], [28, 1084]]
F1_TOL = 0.001
AUC_TOL = 0.001
CELL_TOL = 2

ingest()
stage()
curate()
# The data hash is checked inside load_split and stops the run if it is wrong
X_train, X_test, y_train, y_test = load_split()

# Layer 1: exact checks on the data side
layer1 = {
    "train_shape": X_train.shape == (12028, 215),
    "test_shape": X_test.shape == (3008, 215),
    "test_class_counts": (int((y_test == 0).sum()), int((y_test == 1).sum())) == (1896, 1112),
}
lr = LogisticRegression(penalty="l2", solver="liblinear", C=1.0, max_iter=1000, random_state=SEED)
lr.fit(X_train, y_train)
lr_cm = confusion_matrix(y_test, lr.predict(X_test)).tolist()
layer1["lr_control"] = lr_cm == LR_EXPECTED_CM

# Layer 2: tolerance checks on XGBoost
model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                          random_state=SEED, n_jobs=N_JOBS, **PARAMS)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred).tolist()
f1 = float(f1_score(y_test, pred))
auc = float(roc_auc_score(y_test, proba))

layer2 = {
    "f1": abs(f1 - BASELINE_F1) <= F1_TOL,
    "roc_auc": abs(auc - BASELINE_AUC) <= AUC_TOL,
    "cells": all(
        abs(a - b) <= CELL_TOL
        for row_cm, row_base in zip(cm, BASELINE_CM)
        for a, b in zip(row_cm, row_base)
    ),
}

l1_pass = all(layer1.values())
l2_pass = all(layer2.values())

print("Layer 1 (data side):", "PASS" if l1_pass else "FAIL", layer1)
print("Layer 2 (XGBoost):  ", "PASS" if l2_pass else "FAIL", layer2)
print("f1", round(f1, 4), "roc_auc", round(auc, 4), "cm", cm, "n_jobs", N_JOBS)

Path("outputs").mkdir(exist_ok=True)
with open("outputs/reproduction_report.json", "w", encoding="utf-8") as f:
    json.dump({"n_jobs": N_JOBS, "layer1": layer1, "layer2": layer2,
               "f1": f1, "roc_auc": auc, "confusion_matrix": cm,
               "lr_confusion_matrix": lr_cm}, f, indent=2)

if not l1_pass:
    sys.exit(2)
if not l2_pass:
    sys.exit(3)

    