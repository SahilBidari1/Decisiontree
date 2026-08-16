from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

for depth in [6, 8, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(depth, "train:", train_acc, "test:", test_acc, "leaves:", clf.get_n_leaves())