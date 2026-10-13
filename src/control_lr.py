from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.prepare import load_split, SEED

X_train, X_test, y_train, y_test = load_split()

lr = LogisticRegression(penalty="l2", solver="liblinear", C=1.0, max_iter=1000, random_state=SEED)
lr.fit(X_train, y_train)
proba = lr.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("accuracy", round(accuracy_score(y_test, pred), 4))
print("f1", round(f1_score(y_test, pred), 4))
print("roc_auc", round(roc_auc_score(y_test, proba), 4))
print("confusion_matrix", confusion_matrix(y_test, pred).tolist())
print("expected: accuracy 0.9764, f1 0.9683, roc_auc 0.9961, [[1853, 43], [28, 1084]]")
