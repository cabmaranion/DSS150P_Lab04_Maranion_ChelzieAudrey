import json
import os
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
)

from src.prepare import load_split, SEED

N_JOBS = int(os.environ.get("N_JOBS", "1"))

PARAMS = {
    "n_estimators": 276,
    "max_depth": 7,
    "learning_rate": 0.19781412884706517,
    "subsample": 0.8525778405097011,
    "colsample_bytree": 0.7012859229289896,
    "scale_pos_weight": 1.0,
}

BASELINE = {"f1": 0.9865, "roc_auc": 0.9991,
            "confusion_matrix": [[1882, 14], [16, 1096]]}
TOLERANCE = 0.001

X_train, X_test, y_train, y_test = load_split()

model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="auc",
    random_state=SEED, n_jobs=N_JOBS, **PARAMS,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred)),
    "recall": float(recall_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "pr_auc": float(average_precision_score(y_test, proba)),
    "log_loss": float(log_loss(y_test, proba)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}

for name, value in metrics.items():
    print(name, value if name == "confusion_matrix" else round(value, 4))

f1_ok = abs(metrics["f1"] - BASELINE["f1"]) <= TOLERANCE
auc_ok = abs(metrics["roc_auc"] - BASELINE["roc_auc"]) <= TOLERANCE
cm_ok = metrics["confusion_matrix"] == BASELINE["confusion_matrix"]
print("F1 within tolerance:", f1_ok)
print("ROC-AUC within tolerance:", auc_ok)
print("Confusion matrix identical:", cm_ok)

Path("outputs").mkdir(exist_ok=True)
with open(f"outputs/metrics_njobs{N_JOBS}.json", "w", encoding="utf-8") as f:
    json.dump({"n_jobs": N_JOBS, "tolerance": TOLERANCE, "metrics": metrics,
               "baseline": BASELINE, "f1_ok": f1_ok, "auc_ok": auc_ok,
               "confusion_matrix_identical": cm_ok}, f, indent=2)

    