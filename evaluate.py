from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("accuracy: ", accuracy_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("recall:   ", recall_score(y_test, y_pred))
print("f1:       ", f1_score(y_test, y_pred))
print("roc_auc:  ", roc_auc_score(y_test, y_proba))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))