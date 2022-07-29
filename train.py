import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    nthread=4,
    scale_pos_weight=1,
    seed=27,
)
model = xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred), 3)
roc_auc = round(roc_auc_score(y_test, y_pred), 3)
print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")

joblib.dump(model, "models/xgb-fraud-classifier.joblib")
