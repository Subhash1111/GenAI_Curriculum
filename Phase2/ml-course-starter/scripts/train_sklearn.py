#!/usr/bin/env python
"""
Reusable training script for basic ML tasks using scikit-learn.
Examples:
  python scripts/train_sklearn.py --task classification --dataset iris --model random_forest
  python scripts/train_sklearn.py --task regression --dataset diabetes --model linear
"""
import argparse, joblib, json, os
from pathlib import Path
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

MODELS = {
    "classification": {
        "logistic": LogisticRegression(max_iter=1000),
        "svm": SVC(probability=True),
        "random_forest": RandomForestClassifier(),
        "gbt": GradientBoostingClassifier(),
    },
    "regression": {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "svr": SVR(),
        "random_forest": RandomForestRegressor(),
        "gbr": GradientBoostingRegressor(),
    }
}

DATASETS = {
    "classification": {"iris": datasets.load_iris, "breast_cancer": datasets.load_breast_cancer, "wine": datasets.load_wine},
    "regression": {"diabetes": datasets.load_diabetes}
}

def load_xy(task, dataset):
    loader = DATASETS[task][dataset]
    d = loader()
    return d["data"], d["target"], getattr(d, "feature_names", None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    X, y, feature_names = load_xy(args.task, args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if args.task=="classification" else None)

    model = MODELS[args.task][args.model]
    pipe = Pipeline([("scaler", StandardScaler(with_mean=False) if args.model in ["svm","svr"] else StandardScaler()),
                     ("model", model)])
    pipe.fit(X_train, y_train)

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.outdir, f"{args.task}_{args.model}_{args.dataset}.joblib"))

    # metrics
    if args.task == "classification":
        proba_supported = hasattr(pipe.named_steps["model"], "predict_proba")
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        roc = roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class="ovr") if proba_supported else None
        metrics = {"accuracy": acc, "f1_weighted": f1, "roc_auc": roc}
    else:
        y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {"rmse": mse**0.5, "r2": r2}

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
