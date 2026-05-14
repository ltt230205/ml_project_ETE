import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


def load_config() -> dict:
    params_path = os.getenv("PARAMS_PATH", "params_threshold.yaml")

    if not os.path.exists(params_path):
        params_path = "params.yaml"

    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_running_inside_airflow_container() -> bool:
    cwd = str(Path.cwd())

    return (
        os.getenv("AIRFLOW_HOME") is not None
        or cwd.startswith("/opt/airflow")
        or Path("/opt/airflow").exists()
    )


def normalize_url_for_container(url: str | None, service_name: str, port: int) -> str:
    if not url:
        return f"http://localhost:{port}"

    if is_running_inside_airflow_container():
        url = url.replace(f"http://localhost:{port}", f"http://{service_name}:{port}")
        url = url.replace(f"http://127.0.0.1:{port}", f"http://{service_name}:{port}")

    return url


def setup_mlflow(config: dict) -> None:
    mlflow_config = config.get("mlflow", {})

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_config.get("tracking_uri", "http://localhost:5000"),
    )

    s3_endpoint = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL",
        mlflow_config.get("s3_endpoint", "http://localhost:9000"),
    )

    tracking_uri = normalize_url_for_container(
        tracking_uri,
        service_name="mlflow",
        port=5000,
    )

    s3_endpoint = normalize_url_for_container(
        s3_endpoint,
        service_name="minio",
        port=9000,
    )

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    mlflow.set_tracking_uri(tracking_uri)

    print(f"🔗 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"🪣 MLflow S3 endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object"]).columns.tolist()

        for col in self.columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders[col] = le
            else:
                print(f"⚠️ Column '{col}' not found in data")

        return self

    def transform(self, X):
        X_copy = X.copy()

        for col, le in self.encoders.items():
            if col in X_copy.columns:

                def safe_transform(x):
                    x_str = str(x)

                    if x_str in le.classes_:
                        return le.transform([x_str])[0]

                    return -1

                X_copy[col] = X_copy[col].apply(safe_transform)
            else:
                print(f"⚠️ Column '{col}' not found during transform")

        return X_copy


def build_pipeline(model, scaler, sampling: str, categorical_cols: list[str]) -> ImbPipeline:
    steps = [
        ("label_encoder", MultiColumnLabelEncoder(columns=categorical_cols)),
        ("scaler", scaler),
    ]

    if sampling == "smote":
        steps.append(("smote", SMOTE(random_state=42)))

    steps.append(("model", model))

    return ImbPipeline(steps)


def build_models() -> dict[str, Any]:
    return {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }


def search_best_threshold(
    y_true,
    y_prob,
    metric: str = "f1",
    min_precision: float | None = None,
) -> tuple[float, dict]:
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        if min_precision is not None and precision < min_precision:
            continue

        if metric == "recall":
            score = recall
        elif metric == "precision":
            score = precision
        elif metric == "accuracy":
            score = accuracy
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 2))
            best_metrics = {
                "threshold_metric": metric,
                "threshold_score": float(score),
                "threshold_accuracy": float(accuracy),
                "threshold_precision_churned": float(precision),
                "threshold_recall_churned": float(recall),
                "threshold_f1_churned": float(f1),
            }

    if not best_metrics:
        raise RuntimeError(
            "Không tìm được threshold phù hợp. Hãy giảm min_precision hoặc đổi metric."
        )

    return best_threshold, best_metrics


def safe_report_metric(report: dict, label: str, metric: str, default: float = 0.0) -> float:
    return float(report.get(label, {}).get(metric, default))


def set_model_version_threshold_tags(
    registry_name: str,
    run_id: str,
    threshold: float,
    threshold_metrics: dict,
) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    versions = list(client.search_model_versions(f"name='{registry_name}'"))
    matched_versions = [v for v in versions if v.run_id == run_id]

    if not matched_versions:
        print("⚠️ Không tìm thấy model version tương ứng với final run.")
        return

    latest_version = max(matched_versions, key=lambda v: int(v.version))

    client.set_model_version_tag(
        name=registry_name,
        version=latest_version.version,
        key="decision_threshold",
        value=str(threshold),
    )

    for key, value in threshold_metrics.items():
        client.set_model_version_tag(
            name=registry_name,
            version=latest_version.version,
            key=key,
            value=str(value),
        )

    try:
        client.transition_model_version_stage(
            name=registry_name,
            version=latest_version.version,
            stage="Production",
        )
        print(f"🚀 Đã chuyển {registry_name} v{latest_version.version} sang Production")
    except Exception as e:
        print(f"⚠️ Không thể transition stage: {e}")


def train():
    config = load_config()
    setup_mlflow(config)

    train_df = pd.read_csv(config["processed_data"]["train_path"])
    test_df = pd.read_csv(config["processed_data"]["test_path"])
    target = config["preprocess"]["target_column"]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    print(f"📋 Categorical columns sẽ được encode: {categorical_cols}")

    cv_folds = int(config["preprocess"].get("cv_folds", 5))
    sampling_methods = config["preprocess"].get("sampling_methods", ["none", "smote"])

    experiment_name = config["mlflow"]["experiment_name"]
    registry_name = config["mlflow"].get("registry_name", "churn_model_final_threshold")

    threshold_cfg = config.get("threshold", {})

    # Threshold cố định cho bài toán churn
    fixed_threshold = float(threshold_cfg.get("fixed_value", 0.35))
    threshold_metric = "fixed"
    min_precision = None

    scalers = {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler(),
    }

    models = build_models()

    scoring_metrics = {
        "accuracy": "accuracy",
        "f1_c1": make_scorer(f1_score, pos_label=1),
        "f1_c0": make_scorer(f1_score, pos_label=0),
        "recall_c1": make_scorer(recall_score, pos_label=1),
        "recall_c0": make_scorer(recall_score, pos_label=0),
        "precision_c1": make_scorer(precision_score, pos_label=1),
        "precision_c0": make_scorer(precision_score, pos_label=0),
    }

    mlflow.set_experiment(experiment_name)

    candidates = []

    for scaler_name, scaler in scalers.items():
        for sampling in sampling_methods:
            for model_name, model in models.items():
                run_name = f"{model_name}_{scaler_name}_SMOTE_{sampling}"

                pipeline = build_pipeline(
                    model=model,
                    scaler=scaler,
                    sampling=sampling,
                    categorical_cols=categorical_cols,
                )

                with mlflow.start_run(run_name=run_name) as run:
                    cv_results = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cv_folds,
                        scoring=scoring_metrics,
                        n_jobs=-1,
                    )

                    cv_accuracy = float(cv_results["test_accuracy"].mean())
                    cv_f1_churned = float(cv_results["test_f1_c1"].mean())

                    mlflow.log_params(
                        {
                            "model_type": model_name,
                            "scaler_type": scaler_name,
                            "sampling_type": sampling,
                            "categorical_columns": str(categorical_cols),
                            "cv_folds": cv_folds,
                        }
                    )

                    mlflow.log_metrics(
                        {
                            "cv_accuracy": cv_accuracy,
                            "cv_f1_churned": cv_f1_churned,
                            "cv_f1_stayed": float(cv_results["test_f1_c0"].mean()),
                            "cv_recall_churned": float(cv_results["test_recall_c1"].mean()),
                            "cv_recall_stayed": float(cv_results["test_recall_c0"].mean()),
                            "cv_precision_churned": float(cv_results["test_precision_c1"].mean()),
                            "cv_precision_stayed": float(cv_results["test_precision_c0"].mean()),
                        }
                    )

                    pipeline.fit(X_train, y_train)

                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path="model_file",
                        input_example=X_train.head(5),
                    )

                    candidates.append(
                        {
                            "run_id": run.info.run_id,
                            "run_name": run_name,
                            "model_name": model_name,
                            "scaler_name": scaler_name,
                            "sampling": sampling,
                            "cv_accuracy": cv_accuracy,
                            "cv_f1_churned": cv_f1_churned,
                            "pipeline": pipeline,
                        }
                    )

                    print(f"✅ Done CV: {run_name} | Acc: {cv_accuracy:.4f}")

    if not candidates:
        raise RuntimeError("Không có candidate model nào được train.")

    best = sorted(candidates, key=lambda x: x["cv_accuracy"], reverse=True)[0]

    print("\n--- Best candidate ---")
    print(f"🏆 Best model: {best['run_name']}")
    print(f"📊 CV Accuracy: {best['cv_accuracy']:.4f}")

    print("\n--- Fixed threshold evaluation ---")

    best_threshold = fixed_threshold

    best_pipeline_template = clone(best["pipeline"])

    y_oof_prob = cross_val_predict(
        best_pipeline_template,
        X_train,
        y_train,
        cv=cv_folds,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    y_oof_pred = (y_oof_prob >= best_threshold).astype(int)

    threshold_precision = precision_score(y_train, y_oof_pred, zero_division=0)
    threshold_recall = recall_score(y_train, y_oof_pred, zero_division=0)
    threshold_f1 = f1_score(y_train, y_oof_pred, zero_division=0)
    threshold_accuracy = accuracy_score(y_train, y_oof_pred)

    threshold_metrics = {
        "threshold_metric": "fixed",
        "threshold_score": float(threshold_f1),
        "threshold_accuracy": float(threshold_accuracy),
        "threshold_precision_churned": float(threshold_precision),
        "threshold_recall_churned": float(threshold_recall),
        "threshold_f1_churned": float(threshold_f1),
    }

    print(f"🎯 Fixed threshold: {best_threshold}")
    print(f"📌 Threshold metrics: {threshold_metrics}")

    final_pipeline = clone(best["pipeline"])
    final_pipeline.fit(X_train, y_train)

    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    test_metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_f1_churned": safe_report_metric(test_report, "1", "f1-score"),
        "test_f1_stayed": safe_report_metric(test_report, "0", "f1-score"),
        "test_recall_churned": safe_report_metric(test_report, "1", "recall"),
        "test_precision_churned": safe_report_metric(test_report, "1", "precision"),
    }

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    threshold_artifact = {
        "decision_threshold": best_threshold,
        "threshold_selection_metric": threshold_metric,
        "min_precision": min_precision,
        "threshold_metrics_oof": threshold_metrics,
        "best_candidate_run_id": best["run_id"],
        "best_candidate_run_name": best["run_name"],
        "test_metrics": test_metrics,
    }

    threshold_path = artifact_dir / "decision_threshold.json"
    threshold_path.write_text(
        json.dumps(threshold_artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="register_best_model_with_threshold") as final_run:
        mlflow.log_params(
            {
                "best_candidate_run_id": best["run_id"],
                "best_candidate_run_name": best["run_name"],
                "best_model_type": best["model_name"],
                "best_scaler_type": best["scaler_name"],
                "best_sampling_type": best["sampling"],
                "decision_threshold": best_threshold,
                "threshold_metric": threshold_metric,
            }
        )

        mlflow.log_metrics({**threshold_metrics, **test_metrics})

        mlflow.log_artifact(
            str(threshold_path),
            artifact_path="threshold",
        )

        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model_final",
            registered_model_name=registry_name,
            input_example=X_test.head(5),
        )

        print(f"✅ Đã register final model vào Model Registry: {registry_name}")
        print(f"🎯 Decision threshold: {best_threshold}")
        print(classification_report(y_test, y_test_pred))

        set_model_version_threshold_tags(
            registry_name=registry_name,
            run_id=final_run.info.run_id,
            threshold=best_threshold,
            threshold_metrics=threshold_metrics,
        )


if __name__ == "__main__":
    train()