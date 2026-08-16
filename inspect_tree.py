from sklearn.tree import export_text

small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
small_tree.fit(X_train, y_train)
print(export_text(small_tree, feature_names=list(X.columns)))