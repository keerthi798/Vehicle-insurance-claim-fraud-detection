import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier

df_train = pd.read_csv("E:/S2_network/project4/pythonProject3/data/processed/train.csv")
df_val = pd.read_csv("E:/S2_network/project4/pythonProject3/data/processed/val.csv")
df_test = pd.read_csv("E:/S2_network/project4/pythonProject3/data/processed/test.csv")

X_train = df_train.drop(columns=["claim_number", "fraud"])
y_train = df_train["fraud"]
X_val = df_val.drop(columns=["claim_number", "fraud"])
y_val = df_val["fraud"]
X_test = df_test.drop(columns=["claim_number"])

categorical_features = X_train.columns[X_train.dtypes == object].tolist()
column_transformer = make_column_transformer(
    (OneHotEncoder(drop="first"), categorical_features),
    remainder="passthrough",
)
scaler = MinMaxScaler()

def modeling(X_train, y_train, X_val, y_val, steps):
    pipeline = make_pipeline(*steps)
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict_proba(X_val)[:, 1]
    metric = roc_auc_score(y_val, y_val_pred)
    if isinstance(pipeline._final_estimator, RandomizedSearchCV) or isinstance(pipeline._final_estimator, GridSearchCV):
        print(f"Best params: {pipeline._final_estimator.best_params_}")
    print(f"AUC score: {metric}")
    return pipeline
param_grid = {
    "n_neighbors": [5, 10, 25, 50],
    "weights": ["uniform", "distance"],
}

knn_clf = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    cv=5,
    scoring="roc_auc",
)

knn_pipeline = modeling(X_train, y_train, X_val, y_val, [column_transformer, scaler, knn_clf])

lr_clf = LogisticRegression()
lr_pipeline = modeling(X_train, y_train, X_val, y_val, [column_transformer, scaler, lr_clf])

def add_dummies(df, categorical_features):
    dummies = pd.get_dummies(df[categorical_features], drop_first=True)
    df = pd.concat([dummies, df], axis=1)
    df = df.drop(categorical_features, axis=1)
    return df.columns

feature_names = add_dummies(X_train, categorical_features)

coefficients_df = pd.DataFrame({
    "feature_name": feature_names,
    "coefficient": lr_pipeline._final_estimator.coef_[0]
}).sort_values(by="coefficient", ascending=False).reset_index(drop=True)

print(coefficients_df)

param_grid = {
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    "gamma": [0, 0.25, 0.5, 1.0],
    "n_estimators": [10, 20, 40, 60, 80, 100, 150, 200]
}

xgb_clf = RandomizedSearchCV(
    XGBClassifier(),
    param_distributions=param_grid,
    n_iter=50,
    n_jobs=-1,
    cv=5,
    random_state=23,
    scoring="roc_auc",
)

xgb_pipeline = modeling(X_train, y_train, X_val, y_val, [column_transformer, scaler, xgb_clf])


sampler = SMOTE(random_state=42)
xgb_pipeline_smote = modeling(X_train, y_train, X_val, y_val, [column_transformer, scaler, sampler, xgb_clf])

best_model = xgb_pipeline._final_estimator.best_estimator_
steps = [column_transformer, scaler, best_model]
pipeline = make_pipeline(*steps)
y_test_pred = pipeline.predict_proba(X_test)[:, 1]

df = pd.DataFrame({
    "claim_number": df_test["claim_number"],
    "fraud": y_test_pred
})
df.to_csv("E:/S2_network/project4/pythonProject3/data/submission/submission.csv", index=False)

X_train["random_feature"] = np.random.uniform(size=len(X_train))
xgb_clf_random_feature = XGBClassifier(**xgb_pipeline._final_estimator.best_params_)
steps = [column_transformer, scaler, xgb_clf_random_feature]
xgb_pipeline_random_feature = make_pipeline(*steps)
xgb_pipeline_random_feature = xgb_pipeline_random_feature.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "feature_name": list(feature_names) + ["random_feature"],
    "importance": xgb_pipeline_random_feature._final_estimator.feature_importances_
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

print(importance_df)



