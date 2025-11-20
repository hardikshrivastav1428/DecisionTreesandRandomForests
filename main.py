import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/heart.csv")     

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

plt.figure(figsize=(18, 8))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=["0", "1"])
plt.title("Decision Tree Visualization")
plt.show()

dt_tuned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_tuned.fit(X_train, y_train)

y_pred_dt_tuned = dt_tuned.predict(X_test)
print("\nTuned Decision Tree (max_depth=4) Accuracy:", accuracy_score(y_test, y_pred_dt_tuned))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
importances.sort_values(ascending=False).plot(kind="bar")
plt.title("Feature Importance (Random Forest)")
plt.show()

scores = cross_val_score(rf, X, y, cv=5)
print("\nCross-validation scores:", scores)
print("Mean CV Accuracy:", scores.mean())

print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf))