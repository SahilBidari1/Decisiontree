import joblib
import pandas as pd

df = pd.read_csv("bank.csv")
print(df.shape)
print(df.isnull().sum().sum(), "missing values")
print(df["deposit"].value_counts(normalize=True))
y = (df["deposit"] == "yes").astype(int)
X = df.drop(columns=["deposit"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
# ['job', 'marital', 'education', 'default', 'housing',
#  'loan', 'contact', 'month', 'poutcome']

X = pd.get_dummies(X, columns=cat_cols)
print(X.shape)  # (11162, 51) -- 9 categorical columns became 42 binary ones
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(len(X_train), len(X_test))  # 8929 2233
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

for depth in [6, 8, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(depth, "train:", train_acc, "test:", test_acc, "leaves:", clf.get_n_leaves())
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
joblib.dump(model, "model.joblib")
joblib.dump(list(X.columns), "columns.joblib")
joblib.dump((y_test, y_pred, y_proba), "eval_data.joblib")