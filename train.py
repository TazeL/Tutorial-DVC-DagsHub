import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import dagshub

X_train = pd.read_csv("feature_sets/X_train.csv")
X_test = pd.read_csv("feature_sets/X_test.csv")
y_train = pd.read_csv("feature_sets/y_train.csv")
y_test = pd.read_csv("feature_sets/y_test.csv")

with dagshub.dagshub_logger() as logger:
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
    logger.log_hyperparams(model_class=type(xgb).__name__)
    logger.log_hyperparams({'model': xgb.get_params()})

    model = xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 3)
    roc_auc = round(roc_auc_score(y_test, y_pred), 3)
    logger.log_metrics(
        {'accuracy': accuracy}
    )
    logger.log_metrics(
        {'roc_auc': roc_auc}
    )
print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")

joblib.dump(model, "models/xgb-fraud-classifier.joblib")