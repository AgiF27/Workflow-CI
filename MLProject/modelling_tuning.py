"""
Telecom churn prediction model training with XGBoost and Bayesian hyperparameter optimization.
Logs metrics and artifacts using MLflow for CI/CD integration.
"""

import os
import warnings
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import shap

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real


# Global configuration
RANDOM_STATE = 42
N_ITER = 25
CV_SPLITS = 5
RUN_NAME = "XGBoost_BayesOpt_Churn"
EXPERIMENT_NAME = "telecom-churn"


import os

def initialize_mlflow_tracking() -> None:
    tracking_dir = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_preprocessed_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load preprocessed training and test datasets."""
    train = pd.read_csv("train_preprocessed.csv")
    test = pd.read_csv("test_preprocessed.csv")
    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]
    X_test = test.drop(columns=["Churn"])
    y_test = test["Churn"]

    print(f"Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, y_train, X_test, y_test


def define_search_space() -> dict:
    """Define the hyperparameter search space for Bayesian optimization."""
    return {
        "n_estimators": Integer(100, 400),
        "max_depth": Integer(3, 8),
        "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "subsample": Real(0.7, 1.0),
        "colsample_bytree": Real(0.7, 1.0),
        "gamma": Real(0, 0.5),
    }


def create_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Instantiate an XGBoost classifier with class imbalance handling."""
    return XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
    )


def log_metrics_and_artifacts(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Log evaluation metrics and save model interpretation artifacts."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"{name}: {value:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Feature Importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importance.head(15).plot(kind="barh")
    plt.title("Top 15 Feature Importance")
    plt.xlabel("Importance")
    plt.savefig("feature_importance.png", bbox_inches="tight")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # SHAP Waterfall
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=10, show=False)
    plt.title("SHAP Waterfall (Prediction Sample)")
    plt.savefig("shap_waterfall.png", bbox_inches="tight")
    mlflow.log_artifact("shap_waterfall.png")
    plt.close()

    # Classification Report
    report = classification_report(
        y_test, y_pred, target_names=["Tidak Churn", "Churn"]
    )
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")


def main() -> None:
    """Execute the full model training and logging pipeline."""
    print("Initializing MLflow tracking...")
    initialize_mlflow_tracking()

    print("Loading preprocessed data...")
    X_train, y_train, X_test, y_test = load_preprocessed_data()

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"scale_pos_weight = {pos_weight:.2f}")

    print("Setting up Bayesian optimization...")
    model = create_model(scale_pos_weight=pos_weight)
    search_space = define_search_space()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        n_iter=N_ITER,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    with mlflow.start_run(run_name=RUN_NAME):
        print("Starting Bayesian optimization...")
        bayes_search.fit(X_train, y_train)

        for param, value in bayes_search.best_params_.items():
            mlflow.log_param(param, value)

        print("Logging metrics and artifacts...")
        log_metrics_and_artifacts(bayes_search.best_estimator_, X_test, y_test)

        input_example = X_train.iloc[0:1]
        mlflow.sklearn.log_model(
            sk_model=bayes_search.best_estimator_,
            artifact_path="model",
            input_example=input_example
        )

        print("Training completed successfully.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()