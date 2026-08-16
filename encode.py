y = (df["deposit"] == "yes").astype(int)
X = df.drop(columns=["deposit"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
# ['job', 'marital', 'education', 'default', 'housing',
#  'loan', 'contact', 'month', 'poutcome']

X = pd.get_dummies(X, columns=cat_cols)
print(X.shape)  # (11162, 51) -- 9 categorical columns became 42 binary ones